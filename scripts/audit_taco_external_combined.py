from pathlib import Path
from collections import Counter
from PIL import Image
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "taco_external_combined"
)

IMAGE_DIR = DATASET_ROOT / "images" / "train"
LABEL_DIR = DATASET_ROOT / "labels" / "train"
YAML_PATH = DATASET_ROOT / "data.yaml"

CLASS_NAMES = [
    "plastic",
    "glass",
    "metal",
    "paper_cardboard",
    "biological",
    "other_trash",
]

NUM_CLASSES = len(CLASS_NAMES)

VALID_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def main():
    print("=" * 70)
    print("WasteWise AI - Combined Dataset Structural Audit")
    print("=" * 70)

    # ---------------------------------------------------------
    # Basic paths
    # ---------------------------------------------------------
    print("\nDataset:")
    print(DATASET_ROOT)

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_ROOT}"
        )

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Image directory not found: {IMAGE_DIR}"
        )

    if not LABEL_DIR.exists():
        raise FileNotFoundError(
            f"Label directory not found: {LABEL_DIR}"
        )

    # ---------------------------------------------------------
    # YAML
    # ---------------------------------------------------------
    print("\n[1] Checking data.yaml")

    if not YAML_PATH.exists():
        raise FileNotFoundError(
            f"Missing data.yaml: {YAML_PATH}"
        )

    with open(YAML_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    yaml_names = config.get("names")

    if isinstance(yaml_names, dict):
        yaml_names = [
            yaml_names[i]
            for i in range(NUM_CLASSES)
        ]

    print("YAML classes:", yaml_names)

    if yaml_names != CLASS_NAMES:
        raise ValueError(
            f"Class mismatch.\n"
            f"Expected: {CLASS_NAMES}\n"
            f"Found:    {yaml_names}"
        )

    if config.get("nc") != NUM_CLASSES:
        raise ValueError(
            f"Expected nc={NUM_CLASSES}, "
            f"found nc={config.get('nc')}"
        )

    print("YAML check: PASS")

    # ---------------------------------------------------------
    # Image / label files
    # ---------------------------------------------------------
    print("\n[2] Collecting files")

    image_files = sorted(
        p
        for p in IMAGE_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in VALID_IMAGE_EXTENSIONS
    )

    label_files = sorted(
        p
        for p in LABEL_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() == ".txt"
    )

    print(f"Images: {len(image_files)}")
    print(f"Labels: {len(label_files)}")

    # ---------------------------------------------------------
    # Matching
    # ---------------------------------------------------------
    print("\n[3] Checking image-label alignment")

    image_stems = {
        p.stem
        for p in image_files
    }

    label_stems = {
        p.stem
        for p in label_files
    }

    missing_labels = sorted(
        image_stems - label_stems
    )

    orphan_labels = sorted(
        label_stems - image_stems
    )

    print(f"Missing labels: {len(missing_labels)}")
    print(f"Orphan labels: {len(orphan_labels)}")

    if missing_labels:
        print("\nFirst missing labels:")
        for item in missing_labels[:10]:
            print(" ", item)

    if orphan_labels:
        print("\nFirst orphan labels:")
        for item in orphan_labels[:10]:
            print(" ", item)

    # ---------------------------------------------------------
    # Annotation validation
    # ---------------------------------------------------------
    print("\n[4] Validating YOLO annotations")

    class_counts = Counter()

    invalid_lines = []
    invalid_class_ids = []
    invalid_coordinates = []
    invalid_box_sizes = []
    empty_labels = []

    total_boxes = 0

    for label_path in label_files:

        lines = label_path.read_text(
            encoding="utf-8"
        ).splitlines()

        usable_lines = [
            line.strip()
            for line in lines
            if line.strip()
        ]

        if not usable_lines:
            empty_labels.append(
                label_path.name
            )
            continue

        for line_number, line in enumerate(
            usable_lines,
            start=1,
        ):
            parts = line.split()

            if len(parts) != 5:
                invalid_lines.append(
                    (
                        label_path.name,
                        line_number,
                        line,
                    )
                )
                continue

            try:
                class_id = int(parts[0])

                x = float(parts[1])
                y = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])

            except ValueError:
                invalid_lines.append(
                    (
                        label_path.name,
                        line_number,
                        line,
                    )
                )
                continue

            if not 0 <= class_id < NUM_CLASSES:
                invalid_class_ids.append(
                    (
                        label_path.name,
                        line_number,
                        class_id,
                    )
                )
                continue

            if not (
                0.0 <= x <= 1.0
                and 0.0 <= y <= 1.0
                and 0.0 <= w <= 1.0
                and 0.0 <= h <= 1.0
            ):
                invalid_coordinates.append(
                    (
                        label_path.name,
                        line_number,
                        line,
                    )
                )
                continue

            if w <= 0.0 or h <= 0.0:
                invalid_box_sizes.append(
                    (
                        label_path.name,
                        line_number,
                        line,
                    )
                )
                continue

            x_min = x - w / 2
            y_min = y - h / 2
            x_max = x + w / 2
            y_max = y + h / 2

            tolerance = 1e-5

            if (
                x_min < -tolerance
                or y_min < -tolerance
                or x_max > 1 + tolerance
                or y_max > 1 + tolerance
            ):
                invalid_coordinates.append(
                    (
                        label_path.name,
                        line_number,
                        line,
                    )
                )
                continue

            total_boxes += 1
            class_counts[class_id] += 1

    # ---------------------------------------------------------
    # Image readability
    # ---------------------------------------------------------
    print("\n[5] Checking image readability")

    corrupt_images = []

    for image_path in image_files:
        try:
            with Image.open(image_path) as img:
                img.verify()
        except Exception as exc:
            corrupt_images.append(
                (
                    image_path.name,
                    str(exc),
                )
            )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)

    print(f"Images                 : {len(image_files)}")
    print(f"Labels                 : {len(label_files)}")
    print(f"Total valid boxes      : {total_boxes}")

    print(
        f"Missing labels        : {len(missing_labels)}"
    )
    print(
        f"Orphan labels         : {len(orphan_labels)}"
    )
    print(
        f"Empty label files     : {len(empty_labels)}"
    )
    print(
        f"Invalid annotation lines: {len(invalid_lines)}"
    )
    print(
        f"Invalid class IDs     : {len(invalid_class_ids)}"
    )
    print(
        f"Invalid coordinates   : {len(invalid_coordinates)}"
    )
    print(
        f"Invalid box sizes     : {len(invalid_box_sizes)}"
    )
    print(
        f"Corrupt images        : {len(corrupt_images)}"
    )

    print("\nClass distribution:")

    for class_id, class_name in enumerate(CLASS_NAMES):
        count = class_counts[class_id]
        print(
            f"{class_id}: "
            f"{class_name:<18} "
            f"{count}"
        )

    # ---------------------------------------------------------
    # Expected counts
    # ---------------------------------------------------------
    expected_images = 8523
    expected_boxes = 55510

    print("\nExpected:")
    print(f"Images: {expected_images}")
    print(f"Boxes : {expected_boxes}")

    image_count_ok = (
        len(image_files) == expected_images
    )

    box_count_ok = (
        total_boxes == expected_boxes
    )

    structural_ok = all(
        [
            image_count_ok,
            box_count_ok,
            len(image_files) == len(label_files),
            len(missing_labels) == 0,
            len(orphan_labels) == 0,
            len(empty_labels) == 0,
            len(invalid_lines) == 0,
            len(invalid_class_ids) == 0,
            len(invalid_coordinates) == 0,
            len(invalid_box_sizes) == 0,
            len(corrupt_images) == 0,
        ]
    )

    print("\n" + "=" * 70)

    if structural_ok:
        print("STRUCTURAL AUDIT: PASS")
    else:
        print("STRUCTURAL AUDIT: FAIL")

    print("=" * 70)


if __name__ == "__main__":
    main()