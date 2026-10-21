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

    # ---------------------------------------------------------
    # Detector
    #
    # Currently using E3 combined YOLO11s.
    # This can be changed to E4 after evaluation.
    # ---------------------------------------------------------

    DEFAULT_YOLO_MODEL = (
        PROJECT_ROOT
        / "runs"
        / "detect"
        / "results"
        / "taco_external"
        / "yolo11s_combined_baseline-2"
        / "weights"
        / "best.pt"
    )

    # ---------------------------------------------------------
    # Expanded ResNet50
    # ---------------------------------------------------------

    DEFAULT_RESNET_MODEL = (
        PROJECT_ROOT
        / "models"
        / "expanded_resnet50_stage2_best.pth"
    )

    # ---------------------------------------------------------
    # Expanded OOD calibration
    # ---------------------------------------------------------

    DEFAULT_OOD_CALIBRATION = (
        PROJECT_ROOT
        / "models"
        / "expanded_ood_calibration.json"
    )

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
        detection_confidence=0.25,
        image_size=640,
    ):

        # -----------------------------------------------------
        # Device
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # Validate files
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
        # Load YOLO
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
        # Load Expanded ResNet50
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
        # Load Expanded OOD
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
        image_path,
    ):

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

        # =====================================================
        # STEP 1 - YOLO DETECTION
        # =====================================================

        results = self.detector.predict(
            source=image,
            conf=self.detection_confidence,
            imgsz=self.image_size,
            device=0 if self.device.type == "cuda"
            else "cpu",
            verbose=False,
        )

        result = results[0]

        objects = []

        # -----------------------------------------------------
        # No detections
        # -----------------------------------------------------

        if result.boxes is None or len(
            result.boxes
        ) == 0:

            return {

                "success": True,

                "image": {
                    "width": width,
                    "height": height,
                },

                "detector": {
                    "model": str(
                        self.yolo_model_path
                    ),
                    "confidence_threshold": (
                        self.detection_confidence
                    ),
                    "detections": 0,
                },

                "objects": [],

                "summary": {
                    "total_objects": 0,
                    "supported_objects": 0,
                    "unsupported_objects": 0,
                },

                "message": (
                    "No supported waste objects "
                    "were detected in the image."
                ),
            }

        # =====================================================
        # STEP 2 - PROCESS EVERY DETECTION
        # =====================================================

        for object_id, box in enumerate(
            result.boxes,
            start=1,
        ):

            # -------------------------------------------------
            # Bounding box
            # -------------------------------------------------

            coordinates = (
                box.xyxy[0]
                .detach()
                .cpu()
                .tolist()
            )

            x1, y1, x2, y2 = coordinates

            # Clamp to image dimensions

            x1 = max(
                0,
                min(
                    int(round(x1)),
                    width - 1,
                ),
            )

            y1 = max(
                0,
                min(
                    int(round(y1)),
                    height - 1,
                ),
            )

            x2 = max(
                x1 + 1,
                min(
                    int(round(x2)),
                    width,
                ),
            )

            y2 = max(
                y1 + 1,
                min(
                    int(round(y2)),
                    height,
                ),
            )

            # -------------------------------------------------
            # Detector class
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
            # Crop
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
            # Store result
            # -------------------------------------------------

            objects.append({

                "object_id": object_id,

                "bounding_box": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "width": x2 - x1,
                    "height": y2 - y1,
                },

                "detector": {
                    "class": detector_class,
                    "confidence": round(
                        detector_confidence * 100,
                        2,
                    ),
                },

                "classifier": classification,

            })

        # =====================================================
        # SUMMARY
        # =====================================================

        supported_objects = sum(
            1
            for obj in objects
            if obj["classifier"]["is_supported"]
        )

        unsupported_objects = (
            len(objects)
            - supported_objects
        )

        # =====================================================
        # FINAL RESPONSE
        # =====================================================

        return {

            "success": True,

            "image": {
                "width": width,
                "height": height,
            },

            "detector": {

                "model": str(
                    self.yolo_model_path
                ),

                "confidence_threshold": (
                    self.detection_confidence
                ),

                "classes": (
                    self.DETECTOR_CLASSES
                ),

                "detections": len(
                    objects
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
                    self.ood_detector.threshold,
                    4,
                ),
            },

            "objects": objects,

            "summary": {

                "total_objects": len(
                    objects
                ),

                "supported_objects": (
                    supported_objects
                ),

                "unsupported_objects": (
                    unsupported_objects
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