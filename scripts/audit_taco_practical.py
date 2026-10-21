from pathlib import Path
from collections import Counter

ROOT = Path("data/processed/taco_detection_practical")

EXPECTED_IMAGES = 1499
EXPECTED_BOXES = 4782

EXPECTED_CLASSES = [
    "plastic",
    "glass",
    "metal",
    "paper_cardboard",
    "biological",
    "other_trash",
]

def read_yaml_names():
    yaml_path = ROOT / "data.yaml"
    text = yaml_path.read_text(encoding="utf-8")

    names = []
    in_names = False

    for line in text.splitlines():
        stripped = line.strip()

        if stripped == "names:":
            in_names = True
            continue

        if in_names and ":" in stripped:
            left, right = stripped.split(":", 1)
            left = left.strip()
            right = right.strip().strip("'\"")

            if left.isdigit():
                names.append((int(left), right))

    return [name for _, name in sorted(names)]


def main():
    print("=" * 70)
    print("TACO PRACTICAL 6-CLASS DATASET AUDIT")
    print("=" * 70)

    if not ROOT.exists():
        raise RuntimeError(f"Dataset directory not found: {ROOT}")

    names = read_yaml_names()

    print("\nDATASET STRUCTURE")
    print("-" * 70)

    all_errors = []

    for split in ["train", "val", "test"]:
        image_dir = ROOT / "images" / split
        label_dir = ROOT / "labels" / split

        images = sorted(
            p for p in image_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

        labels = sorted(label_dir.glob("*.txt"))

        print(f"{split:5}: {len(images):4} images, {len(labels):4} labels")

        image_stems = {p.stem for p in images}
        label_stems = {p.stem for p in labels}

        missing_labels = image_stems - label_stems
        orphan_labels = label_stems - image_stems

        if missing_labels:
            all_errors.append(
                f"{split}: {len(missing_labels)} images missing labels"
            )

        if orphan_labels:
            all_errors.append(
                f"{split}: {len(orphan_labels)} orphan labels"
            )

    total_images = sum(
        len([
            p for p in (ROOT / "images" / split).iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ])
        for split in ["train", "val", "test"]
    )

    print("\nCLASS CONFIGURATION")
    print("-" * 70)

    print(f"Classes in data.yaml: {len(names)}")

    for i, name in enumerate(names):
        print(f"{i:2} {name}")

    if names != EXPECTED_CLASSES:
        all_errors.append(
            f"Unexpected class list: {names}"
        )

    print("\nYOLO LABEL AUDIT")
    print("-" * 70)

    class_counts = Counter()
    total_boxes = 0

    split_stems = {}

    for split in ["train", "val", "test"]:
        label_dir = ROOT / "labels" / split

        stems = set()

        for label_file in label_dir.glob("*.txt"):
            stems.add(label_file.stem)

            text = label_file.read_text(encoding="utf-8").strip()

            if not text:
                continue

            for line_number, line in enumerate(text.splitlines(), start=1):
                parts = line.split()

                if len(parts) != 5:
                    all_errors.append(
                        f"{label_file}:{line_number}: expected 5 values, got {len(parts)}"
                    )
                    continue

                try:
                    class_id = int(parts[0])
                    x, y, w, h = map(float, parts[1:])
                except ValueError:
                    all_errors.append(
                        f"{label_file}:{line_number}: non-numeric YOLO values"
                    )
                    continue

                if class_id < 0 or class_id >= len(names):
                    all_errors.append(
                        f"{label_file}:{line_number}: invalid class id {class_id}"
                    )
                    continue

                values = [x, y, w, h]

                if not all(0.0 <= value <= 1.0 for value in values):
                    all_errors.append(
                        f"{label_file}:{line_number}: coordinate outside [0,1]"
                    )

                if w <= 0 or h <= 0:
                    all_errors.append(
                        f"{label_file}:{line_number}: non-positive width/height"
                    )

                class_counts[class_id] += 1
                total_boxes += 1

        split_stems[split] = stems

    print(f"Total images: {total_images}")
    print(f"Total boxes:  {total_boxes}")

    print("\nCLASS DISTRIBUTION")
    print("-" * 70)

    for class_id, name in enumerate(names):
        print(
            f"{class_id:2} {name:18} "
            f"{class_counts[class_id]:5}"
        )

    print("\nSPLIT LEAKAGE AUDIT")
    print("-" * 70)

    train_val = split_stems["train"] & split_stems["val"]
    train_test = split_stems["train"] & split_stems["test"]
    val_test = split_stems["val"] & split_stems["test"]

    print(f"Train ∩ Val : {len(train_val)}")
    print(f"Train ∩ Test: {len(train_test)}")
    print(f"Val ∩ Test  : {len(val_test)}")

    if train_val:
        all_errors.append("Train/Val leakage detected")

    if train_test:
        all_errors.append("Train/Test leakage detected")

    if val_test:
        all_errors.append("Val/Test leakage detected")

    print("\nEXPECTED COUNTS")
    print("-" * 70)
    print(f"Expected images: {EXPECTED_IMAGES}")
    print(f"Actual images:   {total_images}")
    print(f"Expected boxes:  {EXPECTED_BOXES}")
    print(f"Actual boxes:    {total_boxes}")

    if total_images != EXPECTED_IMAGES:
        all_errors.append(
            f"Expected {EXPECTED_IMAGES} images, found {total_images}"
        )

    if total_boxes != EXPECTED_BOXES:
        all_errors.append(
            f"Expected {EXPECTED_BOXES} boxes, found {total_boxes}"
        )

    print("\n" + "=" * 70)

    if all_errors:
        print("AUDIT FAILED")
        print("=" * 70)

        for error in all_errors:
            print(f"ERROR: {error}")

        raise SystemExit(1)

    print("AUDIT RESULT")
    print("=" * 70)
    print("PASS - practical dataset is structurally valid.")
    print(f"PASS - {total_images} images verified.")
    print(f"PASS - {total_boxes} boxes verified.")
    print("PASS - YOLO coordinates valid.")
    print("PASS - no missing/orphan labels.")
    print("PASS - no cross-split leakage.")
    print("PASS - 6-class taxonomy verified.")


if __name__ == "__main__":
    main()
