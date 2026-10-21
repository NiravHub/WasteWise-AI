"""
WasteWise AI - End-to-End Offline Inference

Pipeline:
    Raw image
        -> YOLO11n practical detector
        -> detected object crops
        -> Expanded ResNet50 Stage 2
        -> OOD check
        -> annotated image + JSON
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ------------------------------------------------------------
# Project root
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision import transforms
from ultralytics import YOLO

from src.models.resnet50_model import WasteResNet50
from src.inference.ood_detector import OODDetector


# ------------------------------------------------------------
# Model paths
# ------------------------------------------------------------

YOLO_MODEL = (
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "runs"
    / "detect"
    / "results"
    / "taco_practical"
    / "yolo11n_baseline"
    / "weights"
    / "best.pt"
)

RESNET_MODEL = (
    PROJECT_ROOT
    / "models"
    / "expanded_resnet50_stage2_best.pth"
)

OOD_MODEL = (
    PROJECT_ROOT
    / "models" / "expanded_ood_calibration.json"
)


# ------------------------------------------------------------
# Exact Expanded ResNet50 class order
# ------------------------------------------------------------

CLASS_NAMES = [
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


# ------------------------------------------------------------
# YOLO11n practical 6-class taxonomy
# ------------------------------------------------------------

YOLO_CLASS_NAMES = [
    "plastic",
    "glass",
    "metal",
    "paper_cardboard",
    "biological",
    "other_trash",
]


# ------------------------------------------------------------
# ResNet preprocessing
# ------------------------------------------------------------

RESNET_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


# ------------------------------------------------------------
# Model loading
# ------------------------------------------------------------

def load_resnet(device: torch.device) -> WasteResNet50:
    """Load the trained 19-class Expanded ResNet50."""

    model = WasteResNet50(
        num_classes=len(CLASS_NAMES),
        pretrained=False,
    )

    checkpoint = torch.load(
        RESNET_MODEL,
        map_location=device,
    )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model


# ------------------------------------------------------------
# Crop classification
# ------------------------------------------------------------

def classify_crop(
    model: WasteResNet50,
    crop: Image.Image,
    device: torch.device,
    ood_detector: OODDetector,
) -> dict:
    """Run Expanded ResNet50 + OOD detection on one crop."""

    crop = crop.convert("RGB")

    tensor = RESNET_TRANSFORM(crop).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model.extract_features(tensor)
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)

    probs = probabilities[0]

    confidence, class_index = torch.max(probs, dim=0)

    class_index = int(class_index.item())
    confidence = float(confidence.item())

    top_k = min(3, len(CLASS_NAMES))

    top_probs, top_indices = torch.topk(
        probs,
        k=top_k,
    )

    top_predictions = []

    for probability, index in zip(
        top_probs,
        top_indices,
    ):
        top_predictions.append(
            {
                "class": CLASS_NAMES[int(index.item())],
                "confidence": round(
                    float(probability.item()),
                    6,
                ),
            }
        )

    feature_vector = features[0].detach().cpu()

    ood_result = ood_detector.detect(
        feature_vector
    )

    return {
        "class": CLASS_NAMES[class_index],
        "class_index": class_index,
        "confidence": round(
            confidence,
            6,
        ),
        "top3": top_predictions,
        "ood": {
            "is_supported": bool(
                ood_result["is_supported"]
            ),
            "best_class": ood_result["best_class"],
            "similarity": round(
                float(ood_result["similarity"]),
                6,
            ),
            "threshold": round(
                float(ood_result["threshold"]),
                6,
            ),
        },
    }


# ------------------------------------------------------------
# Drawing
# ------------------------------------------------------------

def draw_label(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    text: str,
) -> None:
    """Draw readable text above a bounding box."""

    x, y = position

    try:
        font = ImageFont.truetype(
            "arial.ttf",
            18,
        )
    except OSError:
        font = ImageFont.load_default()

    text_box = draw.textbbox(
        (x, y),
        text,
        font=font,
    )

    padding = 4

    draw.rectangle(
        [
            text_box[0] - padding,
            text_box[1] - padding,
            text_box[2] + padding,
            text_box[3] + padding,
        ],
        fill="black",
    )

    draw.text(
        (x, y),
        text,
        fill="white",
        font=font,
    )


def annotate_image(
    image: Image.Image,
    detections: list[dict],
) -> Image.Image:
    """Draw detection and classification results."""

    output = image.copy().convert("RGB")

    draw = ImageDraw.Draw(output)

    for detection in detections:

        x1, y1, x2, y2 = detection["bbox"]

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=4,
        )

        classifier = detection.get(
            "classifier"
        )

        if classifier:
            label = (
                f'{classifier["class"]} '
                f'{classifier["confidence"] * 100:.1f}%'
            )
        else:
            label = (
                f'{detection["detector_class"]} '
                f'{detection["detector_confidence"] * 100:.1f}%'
            )

        draw_label(
            draw,
            (
                x1,
                max(0, y1 - 25),
            ),
            label,
        )

    return output


# ------------------------------------------------------------
# Main inference pipeline
# ------------------------------------------------------------

def run_inference(
    image_path: Path,
    output_dir: Path,
    detector_confidence: float = 0.25,
) -> dict:

    if not image_path.exists():
        raise FileNotFoundError(
            f"Input image not found: {image_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("=" * 70)
    print("WASTEWISE AI - END-TO-END INFERENCE")
    print("=" * 70)

    print(f"Device:       {device}")
    print(f"Input:        {image_path}")
    print(f"YOLO model:   {YOLO_MODEL}")
    print(f"ResNet model: {RESNET_MODEL}")
    print()

    # --------------------------------------------------------
    # Load and normalize image
    # --------------------------------------------------------

    image = Image.open(image_path)

    image = ImageOps.exif_transpose(
        image
    ).convert("RGB")

    width, height = image.size

    print(
        f"Image size:   {width} x {height}"
    )

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    print()
    print("Loading YOLO11n detector...")

    detector = YOLO(
        str(YOLO_MODEL)
    )

    print("Loading Expanded ResNet50...")

    classifier = load_resnet(
        device
    )

    print("Loading OOD detector...")

    ood_detector = OODDetector(
        calibration_path=str(OOD_MODEL)
    )

    print("Models loaded.")

    # --------------------------------------------------------
    # YOLO detection
    # --------------------------------------------------------

    print()
    print("Running YOLO detection...")

    results = detector.predict(
        source=str(image_path),
        conf=detector_confidence,
        imgsz=640,
        device=(
            0
            if device.type == "cuda"
            else "cpu"
        ),
        verbose=False,
    )

    result = results[0]

    detections = []

    if (
        result.boxes is None
        or len(result.boxes) == 0
    ):
        print("No objects detected.")

    else:

        boxes = result.boxes

        for detection_index in range(
            len(boxes)
        ):

            coordinates = (
                boxes.xyxy[
                    detection_index
                ]
                .cpu()
                .numpy()
            )

            x1, y1, x2, y2 = (
                coordinates.tolist()
            )

            detector_conf = float(
                boxes.conf[
                    detection_index
                ].item()
            )

            detector_class_index = int(
                boxes.cls[
                    detection_index
                ].item()
            )

            detector_class = (
                YOLO_CLASS_NAMES[
                    detector_class_index
                ]
            )

            # ----------------------------------------------
            # Clamp coordinates
            # ----------------------------------------------

            x1 = max(
                0,
                min(width, x1),
            )

            y1 = max(
                0,
                min(height, y1),
            )

            x2 = max(
                0,
                min(width, x2),
            )

            y2 = max(
                0,
                min(height, y2),
            )

            if (
                x2 <= x1
                or y2 <= y1
            ):
                continue

            # ----------------------------------------------
            # Crop
            # ----------------------------------------------

            crop = image.crop(
                (
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2),
                )
            )

            # ----------------------------------------------
            # ResNet classification
            # ----------------------------------------------

            classifier_result = (
                classify_crop(
                    model=classifier,
                    crop=crop,
                    device=device,
                    ood_detector=ood_detector,
                )
            )

            detections.append(
                {
                    "id": (
                        detection_index
                        + 1
                    ),
                    "bbox": [
                        round(x1, 2),
                        round(y1, 2),
                        round(x2, 2),
                        round(y2, 2),
                    ],
                    "detector_class": (
                        detector_class
                    ),
                    "detector_confidence": round(
                        detector_conf,
                        6,
                    ),
                    "classifier": (
                        classifier_result
                    ),
                }
            )

    # --------------------------------------------------------
    # Save annotated image
    # --------------------------------------------------------

    annotated = annotate_image(
        image=image,
        detections=detections,
    )

    output_image = (
        output_dir
        / f"{image_path.stem}_annotated.jpg"
    )

    annotated.save(
        output_image,
        quality=95,
    )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    output_json = (
        output_dir
        / f"{image_path.stem}_results.json"
    )

    result_payload = {
        "image": image_path.name,
        "image_size": {
            "width": width,
            "height": height,
        },
        "device": str(device),
        "detector": {
            "model": str(YOLO_MODEL),
            "taxonomy": YOLO_CLASS_NAMES,
            "confidence_threshold": (
                detector_confidence
            ),
        },
        "classifier": {
            "model": str(RESNET_MODEL),
            "taxonomy": CLASS_NAMES,
        },
        "detections": detections,
        "summary": {
            "objects_detected": len(
                detections
            ),
        },
        "output_image": str(
            output_image
        ),
    }

    with open(
        output_json,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result_payload,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("INFERENCE COMPLETE")
    print("=" * 70)

    print(
        f"Objects detected: {len(detections)}"
    )

    print(
        f"Annotated image:  {output_image}"
    )

    print(
        f"JSON result:      {output_json}"
    )

    print()

    for detection in detections:

        classifier_result = (
            detection["classifier"]
        )

        print(
            f'#{detection["id"]} '
            f'{detection["detector_class"]} '
            f'({detection["detector_confidence"]:.3f}) '
            f'-> '
            f'{classifier_result["class"]} '
            f'({classifier_result["confidence"]:.3f}) '
            f'| '
            f'OOD={classifier_result["ood"]["is_supported"]}'
        )

    print("=" * 70)

    return result_payload


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "WasteWise AI end-to-end inference"
        )
    )

    parser.add_argument(
        "image",
        type=str,
        help="Path to input image",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/inference",
        help=(
            "Directory for output files"
        ),
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help=(
            "YOLO detection confidence threshold"
        ),
    )

    args = parser.parse_args()

    run_inference(
        image_path=Path(
            args.image
        ),
        output_dir=(
            PROJECT_ROOT
            / args.output_dir
        ),
        detector_confidence=(
            args.conf
        ),
    )


if __name__ == "__main__":
    main()



