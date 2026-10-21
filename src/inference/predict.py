"""
WasteWise AI - End-to-End Production Inference Pipeline

Pipeline:

1. YOLO11s detects practical waste objects
2. Each detected object is cropped
3. Expanded ResNet50 classifies the crop into 19 detailed classes
4. Expanded OOD detector checks whether the crop belongs
   to the supported waste feature space
5. Detector/classifier semantic compatibility is checked
6. Disposal guidance is generated for accepted predictions
7. Rejected detections are reported separately
8. Final accounting guarantees:

       RAW DETECTIONS
       =
       ACCEPTED
       + OOD REJECTED
       + COMPATIBILITY REJECTED
       + OTHER REJECTED
"""

from pathlib import Path
import json
import sys
import time

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms
from ultralytics import YOLO

from src.models.resnet50_model import WasteResNet50
from src.inference.ood_detector import OODDetector


class WasteWisePredictor:

    # =========================================================
    # PROJECT ROOT
    # =========================================================

    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    # =========================================================
    # FINAL E4 YOLO11s DETECTOR
    #
    # TACO TRAIN + External TRAIN
    # YOLO11s
    # cls_pw = 0.5
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
    # EXPANDED RESNET50 CLASSIFIER
    # =========================================================

    DEFAULT_RESNET_MODEL = (
        PROJECT_ROOT
        / "models"
        / "expanded_resnet50_stage2_best.pth"
    )

    # =========================================================
    # EXPANDED OOD CALIBRATION
    # =========================================================

    DEFAULT_OOD_CALIBRATION = (
        PROJECT_ROOT
        / "models"
        / "expanded_ood_calibration.json"
    )

    # =========================================================
    # YOLO DETECTOR TAXONOMY
    # =========================================================

    DETECTOR_CLASSES = [
        "plastic",
        "glass",
        "metal",
        "paper_cardboard",
        "biological",
        "other_trash",
    ]

    # =========================================================
    # EXPANDED RESNET50 TAXONOMY
    # =========================================================

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
    # DETECTOR -> CLASSIFIER SEMANTIC COMPATIBILITY
    #
    # YOLO uses broad practical classes.
    # ResNet uses detailed 19-class categories.
    #
    # An object is accepted only when:
    #
    #   1. OOD says it is supported
    #   2. ResNet prediction belongs to the compatible
    #      detailed classes for the YOLO detector class
    # =========================================================

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
    # DISPOSAL GUIDANCE
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
    # CONSTRUCTOR
    # =========================================================

    def __init__(
        self,
        yolo_model_path=None,
        resnet_model_path=None,
        ood_path=None,
        detection_confidence=0.35,
        image_size=640,
    ):

        # -----------------------------------------------------
        # Device
        # -----------------------------------------------------

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
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

        self.image_size = int(
            image_size
        )

        # -----------------------------------------------------
        # Validate model files
        # -----------------------------------------------------

        if not self.yolo_model_path.exists():
            raise FileNotFoundError(
                "YOLO detector checkpoint not found: "
                f"{self.yolo_model_path}"
            )

        if not self.resnet_model_path.exists():
            raise FileNotFoundError(
                "ResNet50 checkpoint not found: "
                f"{self.resnet_model_path}"
            )

        if not self.ood_path.exists():
            raise FileNotFoundError(
                "OOD calibration file not found: "
                f"{self.ood_path}"
            )

        # =====================================================
        # LOAD YOLO
        # =====================================================

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
        # LOAD EXPANDED RESNET50
        # =====================================================

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
        # LOAD EXPANDED OOD DETECTOR
        # =====================================================

        print()
        print("Loading Expanded OOD detector...")

        self.ood_detector = OODDetector(
            calibration_path=str(
                self.ood_path
            )
        )

        print(
            f"OOD threshold: "
            f"{self.ood_detector.threshold}"
        )

        # =====================================================
        # IMAGE PREPROCESSING
        # =====================================================

        self.transform = transforms.Compose([

            transforms.Resize(
                256
            ),

            transforms.CenterCrop(
                224
            ),

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
    # CONFIDENCE LEVEL
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
    # CLASSIFY ONE OBJECT CROP
    # =========================================================

    @torch.no_grad()
    def classify_crop(
        self,
        crop,
        ood_threshold=None,
    ):

        crop = crop.convert(
            "RGB"
        )

        tensor = self.transform(
            crop
        ).unsqueeze(0).to(
            self.device
        )

        # -----------------------------------------------------
        # Feature extraction
        # -----------------------------------------------------

        features = (
            self.classifier.extract_features(
                tensor
            )
        )

        # -----------------------------------------------------
        # OOD detection
        # -----------------------------------------------------

        ood_result = (
            self.ood_detector.detect(
                features,
                threshold=ood_threshold,
            )
        )

        # -----------------------------------------------------
        # Classification
        # -----------------------------------------------------

        logits = self.classifier(
            tensor
        )

        probabilities = F.softmax(
            logits,
            dim=1,
        )[0]

        # -----------------------------------------------------
        # Top-3 predictions
        # -----------------------------------------------------

        top_k = min(
            3,
            len(
                self.CLASSIFIER_CLASSES
            ),
        )

        top_values, top_indices = (
            torch.topk(
                probabilities,
                k=top_k,
            )
        )

        top_predictions = []

        for probability, index in zip(
            top_values,
            top_indices,
        ):

            top_predictions.append({
                "class": self.CLASSIFIER_CLASSES[
                    int(index.item())
                ],

                "confidence": round(
                    float(
                        probability.item()
                    ) * 100,
                    2,
                ),
            })

        # -----------------------------------------------------
        # Best class
        # -----------------------------------------------------

        predicted_index = int(
            torch.argmax(
                probabilities
            ).item()
        )

        predicted_class = (
            self.CLASSIFIER_CLASSES[
                predicted_index
            ]
        )

        confidence = (
            float(
                probabilities[
                    predicted_index
                ].item()
            ) * 100
        )

        # -----------------------------------------------------
        # OOD unsupported object
        # -----------------------------------------------------

        if not ood_result[
            "is_supported"
        ]:

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
                    float(
                        ood_result[
                            "similarity"
                        ]
                    ),
                    4,
                ),

                "ood_threshold": round(
                    float(
                        ood_result[
                            "threshold"
                        ]
                    ),
                    4,
                ),

                "ood_best_class": (
                    ood_result[
                        "best_class"
                    ]
                ),

                "top_predictions": (
                    top_predictions
                ),
            }

        # -----------------------------------------------------
        # Supported object
        # -----------------------------------------------------

        guidance = (
            self.DISPOSAL_GUIDANCE.get(
                predicted_class,
                {
                    "category": "Waste",

                    "recommendation": (
                        "Dispose of this item according "
                        "to the appropriate local waste stream."
                    ),
                },
            )
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
                float(
                    ood_result[
                        "similarity"
                    ]
                ),
                4,
            ),

            "ood_threshold": round(
                float(
                    ood_result[
                        "threshold"
                    ]
                ),
                4,
            ),

            "ood_best_class": (
                ood_result[
                    "best_class"
                ]
            ),

            "top_predictions": (
                top_predictions
            ),
        }

    # =========================================================
    # MAIN PREDICTION
    # =========================================================

    @torch.no_grad()
    def predict(
        self,
        image_path,
        confidence_threshold=None,
        ood_threshold=None,
    ):

        start_time = time.perf_counter()

        effective_confidence_threshold = (
            self.detection_confidence
            if confidence_threshold is None
            else float(confidence_threshold)
        )

        image_path = Path(
            image_path
        )

        if not image_path.exists():

            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        # -----------------------------------------------------
        # Load image
        # -----------------------------------------------------

        image = Image.open(
            image_path
        )

        # Correct EXIF orientation
        image = ImageOps.exif_transpose(
            image
        )

        image = image.convert(
            "RGB"
        )

        width, height = image.size

        image_filename = (
            image_path.name
        )

        # =====================================================
        # STEP 1 - YOLO DETECTION
        # =====================================================

        results = self.detector.predict(

            source=image,

            conf=effective_confidence_threshold,

            imgsz=self.image_size,

            device=(
                0
                if self.device.type == "cuda"
                else "cpu"
            ),

            verbose=False,
        )

        result = results[0]

        # =====================================================
        # INITIALIZE PIPELINE STATE
        # =====================================================

        objects = []

        rejected_objects = []

        total_ood_rejections = 0

        total_compatibility_rejections = 0

        total_other_rejections = 0

        # =====================================================
        # NO DETECTIONS
        # =====================================================

        if (
            result.boxes is None
            or len(result.boxes) == 0
        ):

            duration_ms = round(
                (
                    time.perf_counter()
                    - start_time
                ) * 1000,
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
                        effective_confidence_threshold
                    ),

                    "classes": (
                        self.DETECTOR_CLASSES
                    ),

                    "raw_detections": 0,

                    "accepted_detections": 0,

                    "detections": 0,
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
                        float(
                            self.ood_detector.threshold
                        ),
                        4,
                    ),
                },

                "objects": [],

                "rejected_objects": [],

                "summary": {

                    "raw_detections": 0,

                    "total_objects": 0,

                    "accepted_objects": 0,

                    # Backward-compatible frontend field.
                    # This means accepted supported objects.
                    "supported_objects": 0,

                    # Backward-compatible frontend field.
                    # This means actual OOD-unsupported objects.
                    "unsupported_objects": 0,

                    # Explicit classifier accounting.
                    "classifier_supported_objects": 0,

                    "classifier_unsupported_objects": 0,

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
                    "No waste objects were detected "
                    "above the detector confidence threshold."
                ),
            }

        # =====================================================
        # STEP 2 - PROCESS EVERY DETECTION
        # =====================================================

        raw_detections_count = len(
            result.boxes
        )

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

            x1_raw, y1_raw, x2_raw, y2_raw = (
                coordinates
            )

            # -------------------------------------------------
            # Coordinate validity
            # -------------------------------------------------

            coords_valid = all(

                isinstance(
                    c,
                    (int, float)
                )

                and not (
                    c != c
                )

                for c in (
                    x1_raw,
                    y1_raw,
                    x2_raw,
                    y2_raw,
                )
            )

            # -------------------------------------------------
            # Clamp coordinates
            # -------------------------------------------------

            x1 = max(
                0,
                min(
                    int(
                        round(
                            x1_raw
                        )
                    ),
                    width - 1,
                ),
            )

            y1 = max(
                0,
                min(
                    int(
                        round(
                            y1_raw
                        )
                    ),
                    height - 1,
                ),
            )

            x2 = max(
                x1 + 1,
                min(
                    int(
                        round(
                            x2_raw
                        )
                    ),
                    width,
                ),
            )

            y2 = max(
                y1 + 1,
                min(
                    int(
                        round(
                            y2_raw
                        )
                    ),
                    height,
                ),
            )

            box_width = (
                x2 - x1
            )

            box_height = (
                y2 - y1
            )

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
                < len(
                    self.DETECTOR_CLASSES
                )
            ):

                detector_class = (
                    self.DETECTOR_CLASSES[
                        detector_class_id
                    ]
                )

            else:

                detector_class = (
                    "unknown"
                )

            # =================================================
            # INVALID BOUNDING BOX
            # =================================================

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
                            detector_confidence
                            * 100,
                            2,
                        ),
                    },

                    "classifier": None,

                    "accepted": False,

                    "rejection_reason": (
                        "invalid_crop"
                    ),
                })

                print(
                    "[WASTEWISE] Rejected detection:",
                    f"YOLO={detector_class}",
                    "ResNet=None",
                    (
                        f"YOLO_conf="
                        f"{detector_confidence * 100:.1f}%"
                    ),
                    "OOD=0.0",
                )

                continue

            # =================================================
            # CROP OBJECT
            # =================================================

            crop = image.crop(
                (
                    x1,
                    y1,
                    x2,
                    y2,
                )
            )

            # =================================================
            # DETAILED CLASSIFICATION
            # =================================================

            classification = (
                self.classify_crop(
                    crop,
                    ood_threshold=ood_threshold,
                )
            )

            # =================================================
            # DETECTOR <-> CLASSIFIER VALIDATION
            # =================================================

            predicted_class = (
                classification.get(
                    "prediction"
                )
            )

            is_supported = (
                classification.get(
                    "is_supported",
                    False,
                )
            )

            compatible_classes = (
                self.DETECTOR_CLASS_COMPATIBILITY.get(
                    detector_class,
                    set(),
                )
            )

            is_compatible = (

                is_supported

                and predicted_class
                in compatible_classes
            )

            # =================================================
            # CANDIDATE OBJECT
            # =================================================

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
                        detector_confidence
                        * 100,
                        2,
                    ),
                },

                "classifier": (
                    classification
                ),
            }

            # =================================================
            # REJECTION HANDLING
            # =================================================

            if not is_compatible:

                # -------------------------------------------------
                # OOD rejection
                # -------------------------------------------------

                if not is_supported:

                    rejection_reason = (
                        "ood_rejection"
                    )

                    total_ood_rejections += 1

                # -------------------------------------------------
                # Semantic compatibility rejection
                # -------------------------------------------------

                else:

                    rejection_reason = (
                        "compatibility_mismatch"
                    )

                    total_compatibility_rejections += 1

                candidate_dict[
                    "accepted"
                ] = False

                candidate_dict[
                    "rejection_reason"
                ] = rejection_reason

                rejected_objects.append(
                    candidate_dict
                )

                print(
                    "[WASTEWISE] Rejected detection:",
                    f"YOLO={detector_class}",
                    f"ResNet={predicted_class}",
                    (
                        f"YOLO_conf="
                        f"{detector_confidence * 100:.1f}%"
                    ),
                    (
                        f"OOD="
                        f"{classification.get('ood_similarity')}"
                    ),
                )

                continue

            # =================================================
            # ACCEPTED OBJECT
            # =================================================

            candidate_dict[
                "accepted"
            ] = True

            candidate_dict[
                "rejection_reason"
            ] = None

            objects.append(
                candidate_dict
            )

        # =====================================================
        # SUMMARY & ACCOUNTING
        # =====================================================

        accepted_detections = len(
            objects
        )

        total_rejected = len(
            rejected_objects
        )

        # -----------------------------------------------------
        # Count classifier/OOD support independently from
        # final acceptance.
        #
        # This is important:
        #
        # A crop can be classifier-supported but still rejected
        # because the ResNet class is incompatible with YOLO.
        # -----------------------------------------------------

        classifier_supported_objects = 0

        classifier_unsupported_objects = 0

        for obj in (
            objects + rejected_objects
        ):

            classifier_result = (
                obj.get(
                    "classifier"
                )
            )

            if classifier_result is None:

                # Invalid crop.
                # It does not have a classifier result.
                continue

            if classifier_result.get(
                "is_supported",
                False,
            ):

                classifier_supported_objects += 1

            else:

                classifier_unsupported_objects += 1

        # -----------------------------------------------------
        # Accounting invariant
        #
        # Every raw detection must belong to exactly one
        # terminal state:
        #
        # ACCEPTED
        # OOD REJECTED
        # COMPATIBILITY REJECTED
        # OTHER REJECTED
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
            (
                time.perf_counter()
                - start_time
            ) * 1000,
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
                    effective_confidence_threshold
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

                # Backward-compatible field.
                # This is intentionally RAW YOLO detections.
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
                    float(
                        self.ood_detector.threshold
                        if ood_threshold is None
                        else ood_threshold
                    ),
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

            "rejected_objects": (
                rejected_objects
            ),

            # -------------------------------------------------
            # SUMMARY
            # -------------------------------------------------

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

                # -------------------------------------------------
                # Backward-compatible frontend fields
                # -------------------------------------------------
                #
                # "supported_objects" represents objects that
                # survived the complete pipeline.
                #
                # "unsupported_objects" represents actual OOD
                # rejections, NOT compatibility mismatches.
                # -------------------------------------------------

                "supported_objects": (
                    accepted_detections
                ),

                "unsupported_objects": (
                    classifier_unsupported_objects
                ),

                # -------------------------------------------------
                # Explicit classifier accounting
                # -------------------------------------------------

                "classifier_supported_objects": (
                    classifier_supported_objects
                ),

                "classifier_unsupported_objects": (
                    classifier_unsupported_objects
                ),

                # -------------------------------------------------
                # Rejection accounting
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

    if len(sys.argv) < 2:

        print()
        print(
            "Usage:"
        )

        print(
            "python -m src.inference.predict "
            "<image_path>"
        )

        print()

        sys.exit(1)

    predictor = (
        WasteWisePredictor()
    )

    result = predictor.predict(
        sys.argv[1]
    )

    print()

    print(
        json.dumps(
            result,
            indent=2,
        )
    )