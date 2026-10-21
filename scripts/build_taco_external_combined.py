from pathlib import Path
import shutil
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TACO_ROOT = PROJECT_ROOT / "data" / "processed" / "taco_detection_practical"
EXTERNAL_ROOT = PROJECT_ROOT / "data" / "processed" / "garbage_external_wastewise"

OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "taco_external_combined"

CLASS_NAMES = [
    "plastic",
    "glass",
    "metal",
    "paper_cardboard",
    "biological",
    "other_trash",
]

BOUNDARY_TOLERANCE = 1e-5


def validate_label_line(line, label_path, line_number):
    parts = line.strip().split()

    if len(parts) != 5:
        raise ValueError(
            f"Invalid annotation format in {label_path}, "
            f"line {line_number}: {line.strip()}"
        )

    try:
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
    except ValueError as exc:
        raise ValueError(
            f"Non-numeric annotation in {label_path}, "
            f"line {line_number}: {line.strip()}"
        ) from exc

    if not 0 <= class_id < len(CLASS_NAMES):
        raise ValueError(
            f"Invalid class ID {class_id} in {label_path}, "
            f"line {line_number}. "
            f"Expected 0-{len(CLASS_NAMES) - 1}."
        )

    values = [x_center, y_center, width, height]

    if not all(0.0 <= value <= 1.0 for value in values):
        raise ValueError(
            f"Normalized value outside [0,1] in {label_path}, "
            f"line {line_number}: {line.strip()}"
        )

    if width <= 0.0 or height <= 0.0:
        raise ValueError(
            f"Zero/negative box size in {label_path}, "
            f"line {line_number}: {line.strip()}"
        )

    x_min = x_center - width / 2.0
    y_min = y_center - height / 2.0
    x_max = x_center + width / 2.0
    y_max = y_center + height / 2.0

    if (
        x_min < -BOUNDARY_TOLERANCE
        or y_min < -BOUNDARY_TOLERANCE
        or x_max > 1.0 + BOUNDARY_TOLERANCE
        or y_max > 1.0 + BOUNDARY_TOLERANCE
    ):
        raise ValueError(
            f"Bounding box outside image bounds in {label_path}, "
            f"line {line_number}: {line.strip()}"
        )

    return class_id


def copy_dataset(source_root, split, prefix, counters):
    """
    Copy one dataset split into the combined dataset.

    Expected source layout:

    source_root/
        train/
            images/
            labels/

        valid/
            images/
            labels/

        test/
            images/
            labels/
    """

    if prefix == "taco":
        image_dir = source_root / "images" / split
        label_dir = source_root / "labels" / split
    else:
        image_dir = source_root / split / "images"
        label_dir = source_root / split / "labels"

    if not image_dir.exists():
        raise FileNotFoundError(
            f"Image directory not found: {image_dir}"
        )

    if not label_dir.exists():
        raise FileNotFoundError(
            f"Label directory not found: {label_dir}"
        )

    output_image_dir = OUTPUT_ROOT / "images" / "train"
    output_label_dir = OUTPUT_ROOT / "labels" / "train"

    image_files = sorted(
        [
            p
            for p in image_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in {
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".webp",
            }
        ]
    )

    print(
        f"\n[{prefix.upper()}] Processing source split: {split}"
    )
    print(f"Images found: {len(image_files)}")

    for image_path in image_files:
        label_path = label_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            raise FileNotFoundError(
                f"Missing label for image: {image_path}\n"
                f"Expected: {label_path}"
            )

        output_name = f"{prefix}_{image_path.name}"
        output_image_path = output_image_dir / output_name
        output_label_path = (
            output_label_dir / f"{prefix}_{image_path.stem}.txt"
        )

        shutil.copy2(image_path, output_image_path)

        lines = label_path.read_text(
            encoding="utf-8"
        ).splitlines()

        output_lines = []

        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue

            class_id = validate_label_line(
                line,
                label_path,
                line_number,
            )

            output_lines.append(line.strip())

            counters["boxes"] += 1
            counters["classes"][class_id] += 1

        output_label_path.write_text(
            "\n".join(output_lines)
            + ("\n" if output_lines else ""),
            encoding="utf-8",
        )

        counters["images"] += 1


def main():
    print("=" * 70)
    print("WasteWise AI - TACO + External Combined Dataset Builder")
    print("=" * 70)

    print("\nProject root:")
    print(PROJECT_ROOT)

    print("\nTACO source:")
    print(TACO_ROOT)

    print("\nExternal source:")
    print(EXTERNAL_ROOT)

    print("\nOutput:")
    print(OUTPUT_ROOT)

    if OUTPUT_ROOT.exists():
        raise FileExistsError(
            f"\nOutput directory already exists:\n{OUTPUT_ROOT}\n\n"
            "Delete the incomplete output before rebuilding."
        )

    (OUTPUT_ROOT / "images" / "train").mkdir(
        parents=True,
        exist_ok=True,
    )

    (OUTPUT_ROOT / "labels" / "train").mkdir(
        parents=True,
        exist_ok=True,
    )

    taco_counters = {
        "images": 0,
        "boxes": 0,
        "classes": [0] * len(CLASS_NAMES),
    }

    external_counters = {
        "images": 0,
        "boxes": 0,
        "classes": [0] * len(CLASS_NAMES),
    }

    # ---------------------------------------------------------
    # TACO TRAIN
    # ---------------------------------------------------------
    copy_dataset(
        source_root=TACO_ROOT,
        split="train",
        prefix="taco",
        counters=taco_counters,
    )

    # ---------------------------------------------------------
    # EXTERNAL TRAIN
    # ---------------------------------------------------------
    copy_dataset(
        source_root=EXTERNAL_ROOT,
        split="train",
        prefix="ext",
        counters=external_counters,
    )

    total_images = (
        taco_counters["images"]
        + external_counters["images"]
    )

    total_boxes = (
        taco_counters["boxes"]
        + external_counters["boxes"]
    )

    total_classes = [
        taco_counters["classes"][i]
        + external_counters["classes"][i]
        for i in range(len(CLASS_NAMES))
    ]

    # ---------------------------------------------------------
    # data.yaml
    # ---------------------------------------------------------
    yaml_data = {
        "path": str(OUTPUT_ROOT.resolve()).replace("\\", "/"),
        "train": "images/train",
        "nc": len(CLASS_NAMES),
        "names": {
            i: name
            for i, name in enumerate(CLASS_NAMES)
        },
    }

    yaml_path = OUTPUT_ROOT / "data.yaml"

    yaml_path.write_text(
        yaml.safe_dump(
            yaml_data,
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("BUILD COMPLETE")
    print("=" * 70)

    print("\nTACO TRAIN")
    print(f"Images : {taco_counters['images']}")
    print(f"Boxes  : {taco_counters['boxes']}")

    print("\nEXTERNAL TRAIN")
    print(f"Images : {external_counters['images']}")
    print(f"Boxes  : {external_counters['boxes']}")

    print("\nCOMBINED")
    print(f"Images : {total_images}")
    print(f"Boxes  : {total_boxes}")

    print("\nClass distribution:")
    for class_id, class_name in enumerate(CLASS_NAMES):
        print(
            f"{class_id}: "
            f"{class_name:<18} "
            f"{total_classes[class_id]}"
        )

    print("\nOutput directory:")
    print(OUTPUT_ROOT)

    print("\ndata.yaml:")
    print(yaml_path)

    print("\nDone.")


if __name__ == "__main__":
    main()