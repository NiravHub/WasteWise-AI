from pathlib import Path

from src.inference.predict import WasteWisePredictor


DATASET_DIR = Path("data/raw/garbage-dataset")


def main():

    predictor = WasteWisePredictor()

    print("=" * 80)
    print("WASTEWISE AI - INFERENCE SANITY CHECK")
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
        predicted = result["prediction"]
        confidence = result["confidence"]

        is_correct = actual == predicted

        if is_correct:
            correct += 1

        total += 1

        status = "✓" if is_correct else "✗"

        print(
            f"{status} "
            f"Actual: {actual:12s} | "
            f"Predicted: {predicted:12s} | "
            f"Confidence: {confidence:6.2f}%"
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