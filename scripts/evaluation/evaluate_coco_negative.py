from pathlib import Path
import json
import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT
    / "runs"
    / "detect"
    / "results"
    / "taco_external_balanced"
    / "yolo11s_balanced"
    / "weights"
    / "best.pt"
)

IMAGE_DIR = (
    ROOT
    / "data"
    / "external"
    / "coco"
    / "val2017"
    / "val2017"
)

OUTPUT_PATH = (
    ROOT
    / "runs"
    / "detect"
    / "coco_negative_threshold_results.json"
)

CLASS_NAMES = {
    0: "plastic",
    1: "glass",
    2: "metal",
    3: "paper_cardboard",
    4: "biological",
    5: "other_trash",
}

THRESHOLDS = [0.30, 0.40, 0.50, 0.60]


def evaluate_threshold(model, images, threshold):

    total_images = 0
    images_with_detection = 0
    total_detections = 0

    confidences = []
    detections_by_class = {
        name: 0 for name in CLASS_NAMES.values()
    }

    highest_confidence = []

    for result in model.predict(
        source=str(IMAGE_DIR),
        imgsz=640,
        conf=threshold,
        device=0,
        stream=True,
        verbose=False,
    ):

        total_images += 1

        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            continue

        images_with_detection += 1

        for cls, conf in zip(
            boxes.cls.cpu().numpy(),
            boxes.conf.cpu().numpy(),
        ):

            class_id = int(cls)
            confidence = float(conf)

            total_detections += 1
            confidences.append(confidence)

            class_name = CLASS_NAMES.get(
                class_id,
                f"class_{class_id}",
            )

            if class_name in detections_by_class:
                detections_by_class[class_name] += 1

            highest_confidence.append({
                "image": Path(result.path).name,
                "class": class_name,
                "confidence": confidence,
            })

    highest_confidence.sort(
        key=lambda x: x["confidence"],
        reverse=True,
    )

    image_fp_rate = (
        images_with_detection / total_images
        if total_images else 0
    )

    average_detections = (
        total_detections / total_images
        if total_images else 0
    )

    if confidences:
        stats = {
            "mean": float(np.mean(confidences)),
            "median": float(np.median(confidences)),
            "p95": float(np.percentile(confidences, 95)),
            "max": float(np.max(confidences)),
        }
    else:
        stats = {
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }

    return {
        "threshold": threshold,
        "images": total_images,
        "images_with_detection": images_with_detection,
        "images_without_detection": (
            total_images - images_with_detection
        ),
        "false_positive_image_rate": image_fp_rate,
        "total_detections": total_detections,
        "average_detections_per_image": average_detections,
        "confidence": stats,
        "detections_by_class": detections_by_class,
        "top_20_highest_confidence": highest_confidence[:20],
    }


def main():

    print("=" * 70)
    print("WASTEWISE AI - COCO NEGATIVE THRESHOLD EVALUATION")
    print("=" * 70)

    images = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    print(f"Model: {MODEL_PATH}")
    print(f"Images: {len(images)}")
    print(f"Thresholds: {THRESHOLDS}")
    print()

    model = YOLO(str(MODEL_PATH))

    results = {}

    for threshold in THRESHOLDS:

        print("-" * 70)
        print(f"Testing confidence threshold: {threshold:.2f}")

        result = evaluate_threshold(
            model,
            images,
            threshold,
        )

        results[str(threshold)] = result

        print(
            f"Images with detection: "
            f"{result['images_with_detection']}"
        )

        print(
            f"False-positive image rate: "
            f"{result['false_positive_image_rate'] * 100:.2f}%"
        )

        print(
            f"Total detections: "
            f"{result['total_detections']}"
        )

        print(
            f"Average detections/image: "
            f"{result['average_detections_per_image']:.3f}"
        )

        print(
            f"Mean confidence: "
            f"{result['confidence']['mean']:.4f}"
        )

        print(
            f"Maximum confidence: "
            f"{result['confidence']['max']:.4f}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("FINAL COCO THRESHOLD SUMMARY")
    print("=" * 70)

    print(
        f"{'Threshold':<12}"
        f"{'FP Images':<15}"
        f"{'FP Rate':<15}"
        f"{'Detections':<15}"
        f"{'Det/Image':<15}"
    )

    for threshold in THRESHOLDS:

        r = results[str(threshold)]

        print(
            f"{threshold:<12.2f}"
            f"{r['images_with_detection']:<15}"
            f"{r['false_positive_image_rate'] * 100:<15.2f}"
            f"{r['total_detections']:<15}"
            f"{r['average_detections_per_image']:<15.3f}"
        )

    print()
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()