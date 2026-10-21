import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predict import WasteWisePredictor

DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "garbage-dataset"

E4_MODEL = (
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "results"
    / "taco_external_balanced"
    / "yolo11s_balanced"
    / "weights"
    / "best.pt"
)


def main():

    predictor = WasteWisePredictor(
        yolo_model_path=str(E4_MODEL)
    )

    print("=" * 80)
    print("WASTEWISE AI - E4 INFERENCE SANITY CHECK")
    print("=" * 80)
    print(f"E4 detector: {E4_MODEL}")
    print("=" * 80)

    total = 0
    correct = 0

    for class_dir in sorted(DATASET_DIR.iterdir()):

        if not class_dir.is_dir():
            continue

        images = [
            p for p in class_dir.iterdir()
            if p.suffix.lower() in {
                ".jpg",
                ".jpeg",
                ".png"
            }
        ]

        if not images:
            continue

        image_path = images[0]

        result = predictor.predict(
            str(image_path)
        )

        actual = class_dir.name

        # Current predictor returns object-level detections,
        # so inspect the first supported classification.
        objects = result.get("objects", [])

        predictions = []

        for obj in objects:
            classifier = obj.get("classifier")

            if not classifier:
                continue

            prediction = classifier.get("prediction")

            if prediction:
                predictions.append(
                    (
                        prediction,
                        classifier.get("confidence", 0.0)
                    )
                )

        if predictions:
            predicted, confidence = max(
                predictions,
                key=lambda x: x[1]
            )
        else:
            predicted = None
            confidence = 0.0

        is_correct = actual == predicted

        if is_correct:
            correct += 1

        total += 1

        status = "[OK]" if is_correct else "[MISS]"

        print(
            f"{status} "
            f"Actual: {actual:12s} | "
            f"Predicted: {str(predicted):18s} | "
            f"Confidence: {confidence:6.2f}% | "
            f"Detections: {len(objects)}"
        )

    print("\n" + "=" * 80)
    print(
        f"Correct: {correct}/{total}"
    )

    if total:
        print(
            f"Sanity-check accuracy: "
            f"{correct / total * 100:.2f}%"
        )

    print("=" * 80)


if __name__ == "__main__":
    main()