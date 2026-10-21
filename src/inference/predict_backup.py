"""
WasteWise AI - End-to-End Production Inference Pipeline

Pipeline:
1. YOLO11s detects waste objects
2. Each detected object is cropped
3. Expanded ResNet50 classifies the crop into 19 detailed classes
4. Expanded OOD detector checks whether the crop belongs
   to the supported waste feature space
5. Disposal guidance is generated for supported predictions

Architecture:

Image
   -> YOLO11s detection
   -> object crop
   -> Expanded ResNet50
   -> Expanded OOD
   -> detailed waste prediction
"""

from pathlib import Path
import time
import io

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms
from ultralytics import YOLO

from src.models.resnet50_model import WasteResNet50
from src.inference.ood_detector import OODDetector


class WasteWisePredictor:

    # =========================================================
    # Project paths
    # =========================================================

    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    # =========================================================
    # E4 FINAL YOLO11s DETECTOR
    # TACO TRAIN + External TRAIN
    # Class balancing: cls_pw=0.5
    # =========================================================

    DEFAULT_YOLO_MODEL = (
        PROJECT_ROOT
        / "runs"
        / "detect"
        / "results"
        / "taco_external_balanced"
        / "yolo11s_balanced"
        / "weights"
        / "best.pt"
    )

    # =========================================================
    # Expanded ResNet50 classifier
    # =========================================================

    DEFAULT_RESNET_MODEL = (
        PROJECT_ROOT
        / "models"
        / "expanded_resnet50_stage2_best.pth"
    )

    # =========================================================
    # Expanded OOD calibration
    # =========================================================

    DEFAULT_OOD_CALIBRATION = (
        PROJECT_ROOT
        / "models"
        / "expanded_ood_calibration.json"
    )

    # Optimal operating point empirically determined by TACO detector evaluation sweep
    DEFAULT_DETECTION_CONFIDENCE = 0.35

    # =========================================================
    # Taxonomies
    # =========================================================

    # YOLO practical detector taxonomy
    DETECTOR_CLASSES = [
        "plastic",
        "glass",
        "metal",
        "paper_cardboard",
        "biological",
        "other_trash",
    ]

    # Expanded ResNet50 taxonomy
    CLASSIFIER_CLASSES = [
        "battery",
        "biological",
        "cable",
        "cardboard",
        "clothes",
        "electronic_device",
        "gauze",
        "general_trash",
        "glass",
        "medical_cap",
        "medical_glasses",
        "medical_gloves",
        "metal",
        "paper",
        "plastic",
        "shoe_cover",
        "shoes",
        "test_tube",
        "urine_bag",
    ]

    # =========================================================
    # Detector ↔ Classifier compatibility
    # =========================================================
    #
    # YOLO has 6 coarse classes while ResNet50 has 19
    # detailed classes. A detection is accepted only when
    # the detailed classifier result is compatible with the
    # detector's coarse category.
    #
    DETECTOR_CLASS_COMPATIBILITY = {
        "plastic": {
            "plastic",
            "cable",
            "medical_cap",
            "medical_gloves",
            "shoe_cover",
            "urine_bag",
        },

        "glass": {
            "glass",
            "test_tube",
            "medical_glasses",
        },

        "metal": {
            "metal",
            "battery",
            "electronic_device",
        },

        "paper_cardboard": {
            "paper",
            "cardboard",
        },

        "biological": {
            "biological",
        },

        "other_trash": {
            "general_trash",
            "clothes",
            "shoes",
            "gauze",
            "medical_cap",
            "medical_gloves",
            "shoe_cover",
            "plastic",
            "paper",
            "cardboard",
        },
    }

    # =========================================================
    # Disposal guidance
    # =========================================================

    DISPOSAL_GUIDANCE = {

        "battery": {
            "category": "Hazardous / E-waste",
            "recommendation": (
                "Dispose of batteries at an authorized "
                "battery or e-waste collection point."
            ),
        },

        "biological": {
            "category": "Organic Waste",
            "recommendation": (
                "Place biological waste in the appropriate "
                "organic or composting waste stream."
            ),
        },

        "cable": {
            "category": "E-waste",
            "recommendation": (
                "Send cables to an authorized e-waste "
                "collection or recycling facility."
            ),
        },

        "cardboard": {
            "category": "Recyclable",
            "recommendation": (
                "Keep cardboard clean and dry and place it "
                "in the recyclable waste stream."
            ),
        },

        "clothes": {
            "category": "Reusable / Textile",
            "recommendation": (
                "Donate, reuse, or send textiles to an "
                "appropriate textile recycling facility."
            ),
        },

        "electronic_device": {
            "category": "E-waste",
            "recommendation": (
                "Dispose of electronic devices through an "
                "authorized e-waste collection facility."
            ),
        },

        "gauze": {
            "category": "Medical Waste",
            "recommendation": (
                "Place used gauze in the appropriate "
                "medical or biomedical waste stream."
            ),
        },

        "general_trash": {
            "category": "General Waste",
            "recommendation": (
                "Place non-recyclable residual waste in "
                "the general waste stream."
            ),
        },

        "glass": {
            "category": "Recyclable",
            "recommendation": (
                "Place clean glass in the appropriate "
                "glass recycling stream."
            ),
        },

        "medical_cap": {
            "category": "Medical Waste",
            "recommendation": (
                "Dispose of medical caps through the "
                "appropriate biomedical waste stream."
            ),
        },

        "medical_glasses": {
            "category": "Medical Waste",
            "recommendation": (
                "Dispose of contaminated medical eyewear "
                "through the appropriate medical waste stream."
            ),
        },

        "medical_gloves": {
            "category": "Medical Waste",
            "recommendation": (
                "Dispose of used medical gloves through "
                "the appropriate biomedical waste stream."
            ),
        },

        "metal": {
            "category": "Recyclable",
            "recommendation": (
                "Place metal items in the appropriate "
                "recyclable waste stream."
            ),
        },

        "paper": {
            "category": "Recyclable",
            "recommendation": (
                "Keep paper reasonably clean and dry and "
                "place it in the recyclable waste stream."
            ),
        },

        "plastic": {
            "category": "Recyclable",
            "recommendation": (
                "Clean plastic where appropriate and place "
                "it in the recyclable waste stream."
            ),
        },

        "shoe_cover": {
            "category": "Medical Waste",
            "recommendation": (
                "Used medical shoe covers should be disposed "
                "of through the appropriate medical waste stream."
            ),
        },

        "shoes": {
            "category": "Reusable / Textile",
            "recommendation": (
                "Donate or reuse shoes when possible, or use "
                "an appropriate textile recovery facility."
            ),
        },

        "test_tube": {
            "category": "Medical Waste",
            "recommendation": (
                "Dispose of used test tubes through the "
                "appropriate laboratory or biomedical waste stream."
            ),
        },

        "urine_bag": {
            "category": "Medical Waste",
            "recommendation": (
                "Dispose of used urine bags through the "
                "appropriate biomedical waste stream."
            ),
        },
    }

    # =========================================================
    # Constructor
    # =========================================================

    def __init__(
        self,
        yolo_model_path=None,
        resnet_model_path=None,
        ood_path=None,
        detection_confidence=DEFAULT_DETECTION_CONFIDENCE,
        image_size=640,
        device=None,
        ood_threshold=None,
        detector=None,
        classifier=None,
        ood_detector=None,
    ):

        # -----------------------------------------------------
        # Device
        # -----------------------------------------------------

        if device is not None:
            self.device = (
                torch.device(device)
                if isinstance(device, str)
                else device
            )
        else:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available()
                else "cpu"
            )

        print("=" * 70)
        print("WASTEWISE AI - PRODUCTION INFERENCE")
        print("=" * 70)
        print(f"Device: {self.device}")

        # -----------------------------------------------------
        # Paths
        # -----------------------------------------------------

        self.yolo_model_path = Path(
            yolo_model_path
            if yolo_model_path is not None
            else self.DEFAULT_YOLO_MODEL
        )

        self.resnet_model_path = Path(
            resnet_model_path
            if resnet_model_path is not None
            else self.DEFAULT_RESNET_MODEL
        )

        self.ood_path = Path(
            ood_path
            if ood_path is not None
            else self.DEFAULT_OOD_CALIBRATION
        )

        self.detection_confidence = float(
            detection_confidence
        )

        self.image_size = int(image_size)
        self.ood_threshold_override = (
            float(ood_threshold)
            if ood_threshold is not None
            else None
        )

        # =====================================================
        # Load YOLO
        # =====================================================

        if detector is not None:
            self.detector = detector
            print("YOLO detector provided via dependency injection.")
        else:
            if not self.yolo_model_path.exists():
                raise FileNotFoundError(
                    "YOLO detector checkpoint not found: "
                    f"{self.yolo_model_path}"
                )

            print()
            print("Loading YOLO11s detector...")

            self.detector = YOLO(
                str(self.yolo_model_path)
            )

            print(
                f"YOLO model loaded: "
                f"{self.yolo_model_path}"
            )

        # =====================================================
        # Load Expanded ResNet50
        # =====================================================

        if classifier is not None:
            self.classifier = classifier
            print("ResNet50 classifier provided via dependency injection.")
        else:
            if not self.resnet_model_path.exists():
                raise FileNotFoundError(
                    "ResNet50 checkpoint not found: "
                    f"{self.resnet_model_path}"
                )

            print()
            print("Loading Expanded ResNet50...")

            self.classifier = WasteResNet50(
                num_classes=len(
                    self.CLASSIFIER_CLASSES
                ),
                pretrained=False,
            )

            checkpoint = torch.load(
                self.resnet_model_path,
                map_location=self.device,
                weights_only=False,
            )

            if isinstance(checkpoint, dict):

                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint[
                        "model_state_dict"
                    ]

                elif "state_dict" in checkpoint:
                    state_dict = checkpoint[
                        "state_dict"
                    ]

                else:
                    state_dict = checkpoint

            else:
                state_dict = checkpoint

            self.classifier.load_state_dict(
                state_dict
            )

            self.classifier.to(
                self.device
            )

            self.classifier.eval()

            print(
                f"ResNet50 loaded: "
                f"{self.resnet_model_path}"
            )

        print(
            f"Classifier classes: "
            f"{len(self.CLASSIFIER_CLASSES)}"
        )

        # =====================================================
        # Load Expanded OOD
        # =====================================================

        if ood_detector is not None:
            self.ood_detector = ood_detector
            if self.ood_threshold_override is not None:
                self.ood_detector.threshold = self.ood_threshold_override
            print("OOD detector provided via dependency injection.")
        else:
            if not self.ood_path.exists():
                raise FileNotFoundError(
                    "OOD calibration file not found: "
                    f"{self.ood_path}"
                )

            print()
            print("Loading Expanded OOD detector...")

            self.ood_detector = OODDetector(
                calibration_path=str(
                    self.ood_path
                ),
                threshold=self.ood_threshold_override,
            )

        print(
            f"OOD threshold: "
            f"{self.ood_detector.threshold}"
        )

        # =====================================================
        # Image preprocessing
        # =====================================================

        self.transform = transforms.Compose([

            transforms.Resize(256),

            transforms.CenterCrop(224),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
            ),
        ])

        print()
        print("Pipeline ready.")
        print("=" * 70)

    # =========================================================
    # Confidence level
    # =========================================================

    @staticmethod
    def get_confidence_level(
        confidence
    ):

        if confidence >= 85:
            return "high"

        if confidence >= 60:
            return "moderate"

        return "low"

    # =========================================================
    # Classify object crop
    # =========================================================

    @torch.no_grad()
    def classify_crop(
        self,
        crop,
    ):

        if (
            crop is None
            or not hasattr(crop, "size")
            or crop.size[0] <= 0
            or crop.size[1] <= 0
        ):
            return {
                "is_supported": False,
                "prediction": None,
                "confidence": 0.0,
                "confidence_level": "low",
                "category": "Invalid Crop",
                "recommendation": (
                    "This detected object crop was empty or degenerate."
                ),
                "ood_similarity": 0.0,
                "ood_threshold": round(
                    self.ood_detector.threshold,
                    4,
                ),
                "ood_best_class": None,
                "top_predictions": [],
            }

        tensor = self.transform(
            crop
        ).unsqueeze(0).to(
            self.device
        )

        # -----------------------------------------------------
        # Feature extraction
        # -----------------------------------------------------

        features = self.classifier.extract_features(
            tensor
        )

        # -----------------------------------------------------
        # OOD detection
        # -----------------------------------------------------

        ood_result = self.ood_detector.detect(
            features
        )

        # -----------------------------------------------------
        # Classification (evaluates head directly on extracted features)
        # -----------------------------------------------------

        if hasattr(self.classifier, "backbone") and hasattr(
            self.classifier.backbone, "fc"
        ):
            logits = self.classifier.backbone.fc(features)
        else:
            logits = self.classifier(
                tensor
            )

        probabilities = F.softmax(
            logits,
            dim=1,
        )[0]

        # -----------------------------------------------------
        # Top-3
        # -----------------------------------------------------

        top_k = min(
            3,
            len(self.CLASSIFIER_CLASSES)
        )

        top_values, top_indices = torch.topk(
            probabilities,
            k=top_k,
        )

        top_predictions = []

        for probability, index in zip(
            top_values,
            top_indices,
        ):

            top_predictions.append({

                "class": self.CLASSIFIER_CLASSES[
                    index.item()
                ],

                "confidence": round(
                    probability.item() * 100,
                    2,
                ),
            })

        # -----------------------------------------------------
        # Best class
        # -----------------------------------------------------

        predicted_index = torch.argmax(
            probabilities
        ).item()

        predicted_class = (
            self.CLASSIFIER_CLASSES[
                predicted_index
            ]
        )

        confidence = (
            probabilities[
                predicted_index
            ].item()
            * 100
        )

        # -----------------------------------------------------
        # OOD unsupported object
        # -----------------------------------------------------

        if not ood_result["is_supported"]:

            return {

                "is_supported": False,

                "prediction": None,

                "confidence": round(
                    confidence,
                    2,
                ),

                "confidence_level": "low",

                "category": "Unsupported Image",

                "recommendation": (
                    "This detected object does not appear "
                    "sufficiently similar to the supported "
                    "waste categories."
                ),

                "ood_similarity": round(
                    ood_result["similarity"],
                    4,
                ),

                "ood_threshold": round(
                    ood_result["threshold"],
                    4,
                ),

                "ood_best_class": (
                    ood_result["best_class"]
                ),

                "top_predictions":
                    top_predictions,
            }

        # -----------------------------------------------------
        # Supported object
        # -----------------------------------------------------

        guidance = self.DISPOSAL_GUIDANCE.get(
            predicted_class,
            {
                "category": "Waste",
                "recommendation": (
                    "Dispose of this item according "
                    "to the appropriate local waste stream."
                ),
            },
        )

        return {

            "is_supported": True,

            "prediction": predicted_class,

            "confidence": round(
                confidence,
                2,
            ),

            "confidence_level": (
                self.get_confidence_level(
                    confidence
                )
            ),

            "category": guidance[
                "category"
            ],

            "recommendation": guidance[
                "recommendation"
            ],

            "ood_similarity": round(
                ood_result["similarity"],
                4,
            ),

            "ood_threshold": round(
                ood_result["threshold"],
                4,
            ),

            "ood_best_class": (
                ood_result["best_class"]
            ),

            "top_predictions":
                top_predictions,
        }

    # =========================================================
    # Main prediction
    # =========================================================

    @torch.no_grad()
    def predict(
        self,
        image_input,
        confidence_threshold=None,
        ood_threshold=None,
    ):

        start_time = time.perf_counter()

        # -----------------------------------------------------
        # Load and validate image input
        # -----------------------------------------------------

        image_filename = None

        if isinstance(image_input, (str, Path)):

            image_path = Path(image_input)

            if not image_path.exists():
                raise FileNotFoundError(
                    f"Image not found: {image_path}"
                )

            image_filename = image_path.name
            image = Image.open(image_path)

        elif isinstance(image_input, Image.Image):

            image = image_input
            image_filename = getattr(
                image_input,
                "filename",
                "in_memory.png",
            )

        elif hasattr(image_input, "read"):

            image = Image.open(image_input)
            image_filename = getattr(
                image_input,
                "name",
                "stream.png",
            )

        else:

            raise ValueError(
                f"Unsupported image input type: {type(image_input)}"
            )

        # Correct EXIF orientation
        image = ImageOps.exif_transpose(
            image
        )

        image = image.convert(
            "RGB"
        )

        width, height = image.size

        if width <= 0 or height <= 0:
            raise ValueError(
                f"Invalid image dimensions: {width}x{height}"
            )

        # -----------------------------------------------------
        # Dynamic threshold configuration
        # -----------------------------------------------------

        active_conf = (
            float(confidence_threshold)
            if confidence_threshold is not None
            else self.detection_confidence
        )

        active_ood_threshold = (
            float(ood_threshold)
            if ood_threshold is not None
            else self.ood_detector.threshold
        )

        # =====================================================
        # STEP 1 - YOLO DETECTION
        # =====================================================

        if self.device.type == "cuda":
            yolo_device = (
                self.device.index
                if self.device.index is not None
                else 0
            )
        else:
            yolo_device = "cpu"

        results = self.detector.predict(
            source=image,
            conf=active_conf,
            imgsz=self.image_size,
            device=yolo_device,
            verbose=False,
        )

        result = results[0]

        objects = []
        rejected_objects = []

        total_ood_rejections = 0
        total_compatibility_rejections = 0
        total_other_rejections = 0

        # -----------------------------------------------------
        # No detections
        # -----------------------------------------------------

        if result.boxes is None or len(result.boxes) == 0:

            duration_ms = round(
                (time.perf_counter() - start_time) * 1000,
                2,
            )

            return {

                "success": True,

                "version": "1.0.0",

                "duration_ms": duration_ms,

                "image": {
                    "width": width,
                    "height": height,
                    "filename": image_filename,
                },

                "detector": {
                    "model": str(
                        self.yolo_model_path
                    ),
                    "confidence_threshold": (
                        active_conf
                    ),
                    "detections": 0,
                    "raw_detections": 0,
                    "accepted_detections": 0,
                },

                "classifier": {
                    "model": str(
                        self.resnet_model_path
                    ),
                    "classes": (
                        self.CLASSIFIER_CLASSES
                    ),
                    "class_count": len(
                        self.CLASSIFIER_CLASSES
                    ),
                    "ood_calibration": str(
                        self.ood_path
                    ),
                    "ood_threshold": round(
                        active_ood_threshold,
                        4,
                    ),
                },

                "objects": [],

                "rejected_objects": [],

                "summary": {
                    "raw_detections": 0,
                    "total_objects": 0,
                    "accepted_objects": 0,
                    "supported_objects": 0,
                    "unsupported_objects": 0,
                    "ood_rejections": 0,
                    "compatibility_rejections": 0,
                    "other_rejections": 0,
                    "total_rejections": 0,
                    "accounting_valid": True,
                },

                "accounting": {
                    "raw_detections": 0,
                    "accepted_detections": 0,
                    "ood_rejected": 0,
                    "compatibility_rejected": 0,
                    "other_rejected": 0,
                    "total_rejected": 0,
                    "is_valid": True,
                },

                "message": (
                    "No supported waste objects "
                    "were detected in the image."
                ),
            }

        # =====================================================
        # STEP 2 - PROCESS EVERY DETECTION
        # =====================================================

        raw_detections_count = len(result.boxes)

        for object_id, box in enumerate(
            result.boxes,
            start=1,
        ):

            # -------------------------------------------------
            # Bounding box coordinates
            # -------------------------------------------------

            coordinates = (
                box.xyxy[0]
                .detach()
                .cpu()
                .tolist()
            )

            x1_raw, y1_raw, x2_raw, y2_raw = coordinates

            # Coordinate validity check
            coords_valid = all(
                isinstance(c, (int, float))
                and not (c != c)  # NaN check
                for c in (x1_raw, y1_raw, x2_raw, y2_raw)
            )

            # Clamp coordinates to image boundaries
            x1 = max(
                0,
                min(
                    int(round(x1_raw)),
                    width - 1,
                ),
            )

            y1 = max(
                0,
                min(
                    int(round(y1_raw)),
                    height - 1,
                ),
            )

            x2 = max(
                x1 + 1,
                min(
                    int(round(x2_raw)),
                    width,
                ),
            )

            y2 = max(
                y1 + 1,
                min(
                    int(round(y2_raw)),
                    height,
                ),
            )

            box_width = x2 - x1
            box_height = y2 - y1

            # -------------------------------------------------
            # Detector class and confidence
            # -------------------------------------------------

            detector_class_id = int(
                box.cls[0].item()
            )

            detector_confidence = float(
                box.conf[0].item()
            )

            if (
                0 <= detector_class_id
                < len(self.DETECTOR_CLASSES)
            ):
                detector_class = (
                    self.DETECTOR_CLASSES[
                        detector_class_id
                    ]
                )
            else:
                detector_class = "unknown"

            # -------------------------------------------------
            # Validate bounding box geometry
            # -------------------------------------------------

            if (
                not coords_valid
                or box_width < 2
                or box_height < 2
            ):

                total_other_rejections += 1

                rejected_objects.append({
                    "object_id": object_id,
                    "bounding_box": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "width": box_width,
                        "height": box_height,
                    },
                    "detector": {
                        "class": detector_class,
                        "confidence": round(
                            detector_confidence * 100,
                            2,
                        ),
                    },
                    "classifier": None,
                    "accepted": False,
                    "rejection_reason": "invalid_crop",
                })

                print(
                    "[WASTEWISE] Rejected detection:",
                    f"YOLO={detector_class}",
                    "ResNet=None",
                    f"YOLO_conf={detector_confidence * 100:.1f}%",
                    "OOD=0.0",
                )
                continue

            # -------------------------------------------------
            # Crop object
            # -------------------------------------------------

            crop = image.crop(
                (
                    x1,
                    y1,
                    x2,
                    y2,
                )
            )

            # -------------------------------------------------
            # Detailed classification
            # -------------------------------------------------

            classification = (
                self.classify_crop(
                    crop
                )
            )

            # -------------------------------------------------
            # Detector <-> Classifier validation
            # -------------------------------------------------

            predicted_class = classification.get(
                "prediction"
            )

            is_supported = classification.get(
                "is_supported",
                False,
            )

            compatible_classes = (
                self.DETECTOR_CLASS_COMPATIBILITY.get(
                    detector_class,
                    set(),
                )
            )

            is_compatible = (
                is_supported
                and predicted_class in compatible_classes
            )

            # -------------------------------------------------
            # Candidate structure
            # -------------------------------------------------

            candidate_dict = {
                "object_id": object_id,
                "bounding_box": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "width": box_width,
                    "height": box_height,
                },
                "detector": {
                    "class": detector_class,
                    "confidence": round(
                        detector_confidence * 100,
                        2,
                    ),
                },
                "classifier": classification,
            }

            # -------------------------------------------------
            # Rejection handling
            # -------------------------------------------------

            if not is_compatible:

                if not is_supported:
                    rejection_reason = "ood_rejection"
                    total_ood_rejections += 1
                else:
                    rejection_reason = (
                        "compatibility_mismatch"
                    )
                    total_compatibility_rejections += 1

                candidate_dict["accepted"] = False
                candidate_dict["rejection_reason"] = (
                    rejection_reason
                )

                rejected_objects.append(
                    candidate_dict
                )

                print(
                    "[WASTEWISE] Rejected detection:",
                    f"YOLO={detector_class}",
                    f"ResNet={predicted_class}",
                    f"YOLO_conf={detector_confidence * 100:.1f}%",
                    f"OOD={classification.get('ood_similarity')}",
                )
                continue

            # -------------------------------------------------
            # Store validated accepted result
            # -------------------------------------------------

            candidate_dict["accepted"] = True
            candidate_dict["rejection_reason"] = None

            objects.append(candidate_dict)

                # =====================================================
        # SUMMARY & ACCOUNTING INVARIANT
        # =====================================================

        accepted_detections = len(objects)
        total_rejected = len(rejected_objects)

        # -----------------------------------------------------
        # Classifier support accounting
        #
        # IMPORTANT:
        # "is_supported" means the ResNet/OOD system considers
        # the cropped object to be from a supported distribution.
        #
        # It is NOT the same as "accepted".
        #
        # A classifier-supported object can still be rejected
        # because its class is incompatible with the YOLO class.
        # -----------------------------------------------------

        classifier_supported_objects = 0
        classifier_unsupported_objects = 0

        for obj in objects + rejected_objects:
            classifier_result = obj.get("classifier")

            if classifier_result is None:
                # Invalid crop / processing failure
                # is neither an OOD-supported nor OOD-unsupported
                # classifier result.
                continue

            if classifier_result.get("is_supported", False):
                classifier_supported_objects += 1
            else:
                classifier_unsupported_objects += 1

        # -----------------------------------------------------
        # Accounting invariant
        #
        # Every raw YOLO detection must end up in exactly one of:
        #
        #   1. accepted
        #   2. OOD rejected
        #   3. compatibility rejected
        #   4. other rejected
        # -----------------------------------------------------

        accounting_valid = (
            raw_detections_count
            == (
                accepted_detections
                + total_rejected
            )
        ) and (
            total_rejected
            == (
                total_ood_rejections
                + total_compatibility_rejections
                + total_other_rejections
            )
        )

        duration_ms = round(
            (time.perf_counter() - start_time) * 1000,
            2,
        )

        # =====================================================
        # FINAL RESPONSE
        # =====================================================

        return {

            "success": True,

            "version": "1.0.0",

            "duration_ms": duration_ms,

            # -------------------------------------------------
            # IMAGE
            # -------------------------------------------------

            "image": {
                "width": width,
                "height": height,
                "filename": image_filename,
            },

            # -------------------------------------------------
            # DETECTOR
            # -------------------------------------------------

            "detector": {

                "model": str(
                    self.yolo_model_path
                ),

                "confidence_threshold": (
                    active_conf
                ),

                "classes": (
                    self.DETECTOR_CLASSES
                ),

                "raw_detections": (
                    raw_detections_count
                ),

                "accepted_detections": (
                    accepted_detections
                ),

                # Keep this for backward compatibility.
                # This represents the number of raw YOLO
                # detections before OOD/compatibility filtering.
                "detections": (
                    raw_detections_count
                ),
            },

            # -------------------------------------------------
            # CLASSIFIER
            # -------------------------------------------------

            "classifier": {

                "model": str(
                    self.resnet_model_path
                ),

                "classes": (
                    self.CLASSIFIER_CLASSES
                ),

                "class_count": len(
                    self.CLASSIFIER_CLASSES
                ),

                "ood_calibration": str(
                    self.ood_path
                ),

                "ood_threshold": round(
                    active_ood_threshold,
                    4,
                ),
            },

            # -------------------------------------------------
            # ACCEPTED OBJECTS
            # -------------------------------------------------

            "objects": objects,

            # -------------------------------------------------
            # REJECTED OBJECTS
            # -------------------------------------------------

            "rejected_objects": rejected_objects,

            # -------------------------------------------------
            # SUMMARY
            # -------------------------------------------------

            "summary": {

                # Total YOLO detections entering the pipeline
                "raw_detections": (
                    raw_detections_count
                ),

                # Final objects surviving ALL validation
                "total_objects": (
                    accepted_detections
                ),

                "accepted_objects": (
                    accepted_detections
                ),

                # -------------------------------------------------
                # CLASSIFIER / OOD STATUS
                # -------------------------------------------------
                #
                # These are deliberately separate from
                # accepted_objects.
                #
                # Example:
                #
                # YOLO -> plastic
                # ResNet -> glass
                # OOD -> supported
                #
                # This object is classifier-supported but
                # compatibility-rejected.
                # -------------------------------------------------

                "classifier_supported_objects": (
                    classifier_supported_objects
                ),

                "classifier_unsupported_objects": (
                    classifier_unsupported_objects
                ),

                # -------------------------------------------------
                # REJECTION REASONS
                # -------------------------------------------------

                "ood_rejections": (
                    total_ood_rejections
                ),

                "compatibility_rejections": (
                    total_compatibility_rejections
                ),

                "other_rejections": (
                    total_other_rejections
                ),

                "total_rejections": (
                    total_rejected
                ),

                # -------------------------------------------------
                # FINAL ACCOUNTING CHECK
                # -------------------------------------------------

                "accounting_valid": (
                    accounting_valid
                ),
            },

            # -------------------------------------------------
            # ACCOUNTING
            # -------------------------------------------------

            "accounting": {

                "raw_detections": (
                    raw_detections_count
                ),

                "accepted_detections": (
                    accepted_detections
                ),

                "ood_rejected": (
                    total_ood_rejections
                ),

                "compatibility_rejected": (
                    total_compatibility_rejections
                ),

                "other_rejected": (
                    total_other_rejections
                ),

                "total_rejected": (
                    total_rejected
                ),

                "is_valid": (
                    accounting_valid
                ),
            },
        }


# =============================================================
# CLI TEST
# =============================================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:

        print(
            "Usage:"
        )

        print(
            "python -m src.inference.predict "
            "<image_path>"
        )

        sys.exit(1)

    predictor = WasteWisePredictor()

    result = predictor.predict(
        sys.argv[1]
    )

    import json

    print(
        json.dumps(
            result,
            indent=2,
        )
    )

        # =====================================================
        # FINAL RESPONSE
        # =====================================================

        return {

            "success": True,

            "version": "1.0.0",

            "duration_ms": duration_ms,

            "image": {
                "width": width,
                "height": height,
                "filename": image_filename,
            },

            "detector": {

                "model": str(
                    self.yolo_model_path
                ),

                "confidence_threshold": (
                    active_conf
                ),

                "classes": (
                    self.DETECTOR_CLASSES
                ),

                "raw_detections": (
                    raw_detections_count
                ),

                "accepted_detections": (
                    accepted_detections
                ),

                "detections": (
                    raw_detections_count
                ),
            },

            "classifier": {

                "model": str(
                    self.resnet_model_path
                ),

                "classes": (
                    self.CLASSIFIER_CLASSES
                ),

                "class_count": len(
                    self.CLASSIFIER_CLASSES
                ),

                "ood_calibration": str(
                    self.ood_path
                ),

                "ood_threshold": round(
                    active_ood_threshold,
                    4,
                ),
            },

            "objects": objects,

            "rejected_objects": rejected_objects,

            "summary": {

                "raw_detections": (
                    raw_detections_count
                ),

                "total_objects": (
                    accepted_detections
                ),

                "accepted_objects": (
                    accepted_detections
                ),

                "supported_objects": (
                    accepted_detections
                ),

                "unsupported_objects": (
                    total_rejected
                ),

                "ood_rejections": (
                    total_ood_rejections
                ),

                "compatibility_rejections": (
                    total_compatibility_rejections
                ),

                "other_rejections": (
                    total_other_rejections
                ),

                "total_rejections": (
                    total_rejected
                ),

                "accounting_valid": (
                    accounting_valid
                ),
            },

            "accounting": {
                "raw_detections": raw_detections_count,
                "accepted_detections": accepted_detections,
                "ood_rejected": total_ood_rejections,
                "compatibility_rejected": total_compatibility_rejections,
                "other_rejected": total_other_rejections,
                "total_rejected": total_rejected,
                "is_valid": accounting_valid,
            },
        }


# =============================================================
# CLI TEST
# =============================================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:

        print(
            "Usage:"
        )

        print(
            "python -m src.inference.predict "
            "<image_path>"
        )

        sys.exit(1)

    predictor = WasteWisePredictor()

    result = predictor.predict(
        sys.argv[1]
    )

    import json

    print(
        json.dumps(
            result,
            indent=2,
        )
    )