from pathlib import Path
import csv
import random
import shutil
import yaml
from collections import Counter, defaultdict
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

AUDIT = ROOT / "results" / "dataset_audit" / "taco"
OUTPUT = ROOT / "data" / "processed" / "taco_detection"

ANNOTATIONS = AUDIT / "taco_clean_annotations.csv"
MANIFEST = AUDIT / "taco_image_manifest.csv"

SEED = 42

# ------------------------------------------------------------
# 23 selected TACO classes
# ------------------------------------------------------------

SELECTED = [
    (59, "Cigarette"),
    (58, "Unlabeled litter"),
    (36, "Plastic film"),
    (5, "Clear plastic bottle"),
    (29, "Other plastic"),
    (39, "Other plastic wrapper"),
    (12, "Drink can"),
    (7, "Plastic bottle cap"),
    (55, "Plastic straw"),
    (9, "Broken glass"),
    (57, "Styrofoam piece"),
    (6, "Glass bottle"),
    (21, "Disposable plastic cup"),
    (50, "Pop tab"),
    (14, "Other carton"),
    (33, "Normal paper"),
    (8, "Metal bottle cap"),
    (27, "Plastic lid"),
    (20, "Paper cup"),
    (17, "Corrugated carton"),
    (0, "Aluminium foil"),
    (40, "Single-use carrier bag"),
    (4, "Other plastic bottle"),
]

TACO_TO_YOLO = {
    taco_id: index
    for index, (taco_id, _) in enumerate(SELECTED)
}

NAMES = [name for _, name in SELECTED]

# ------------------------------------------------------------
# Load image manifest
# ------------------------------------------------------------

print("=" * 75)
print("BUILDING TACO YOLO DETECTION DATASET")
print("=" * 75)

image_manifest = {}

with open(MANIFEST, "r", encoding="utf-8") as f:

    for row in csv.DictReader(f):

        image_id = row["image_id"]

        image_path = Path(row["image_path"])

        image_manifest[image_id] = {
            "file_name": row["file_name"],
            "width": int(row["width"]),
            "height": int(row["height"]),
            "image_path": image_path,
        }

print(
    f"\nManifest images: {len(image_manifest):,}"
)

# Verify actual files.

missing_manifest_files = []

for image_id, record in image_manifest.items():

    if not record["image_path"].is_file():

        missing_manifest_files.append(
            (
                image_id,
                record["image_path"]
            )
        )

print(
    f"Missing manifest files: "
    f"{len(missing_manifest_files):,}"
)

if missing_manifest_files:

    print("\nFirst missing files:")

    for image_id, path in missing_manifest_files[:10]:

        print(
            f"  image_id={image_id}: {path}"
        )

    raise RuntimeError(
        "Some manifest image paths do not exist."
    )

# ------------------------------------------------------------
# Load cleaned annotations
# ------------------------------------------------------------

image_records = defaultdict(list)

with open(
    ANNOTATIONS,
    "r",
    encoding="utf-8"
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        taco_category = int(
            row["category_id"]
        )

        if taco_category not in TACO_TO_YOLO:

            continue

        image_id = row["image_id"]

        if image_id not in image_manifest:

            raise RuntimeError(
                f"Annotation references image_id "
                f"{image_id}, but manifest has no such image."
            )

        image_records[image_id].append(row)

print(
    f"Images containing selected classes: "
    f"{len(image_records):,}"
)

print(
    f"Selected annotations: "
    f"{sum(len(v) for v in image_records.values()):,}"
)

# ------------------------------------------------------------
# Deterministic image-level split
#
# Class-aware greedy assignment:
# images are processed by rarity first.
# ------------------------------------------------------------

random.seed(SEED)

images = list(image_records.keys())

# Count image-level class presence.

class_image_counts = Counter()

for image_id in images:

    classes = {
        int(row["category_id"])
        for row in image_records[image_id]
    }

    for category_id in classes:

        class_image_counts[category_id] += 1

# Process images containing rarer classes first.

images.sort(
    key=lambda image_id: (
        min(
            class_image_counts[
                int(row["category_id"])
            ]
            for row in image_records[image_id]
        ),
        -len(image_records[image_id]),
        image_id,
    )
)

target_train = round(
    len(images) * 0.70
)

target_val = round(
    len(images) * 0.15
)

target_test = (
    len(images)
    - target_train
    - target_val
)

target_counts = {
    "train": target_train,
    "val": target_val,
    "test": target_test,
}

splits = {
    "train": [],
    "val": [],
    "test": [],
}

split_class_counts = {
    "train": Counter(),
    "val": Counter(),
    "test": Counter(),
}

split_image_counts = {
    "train": 0,
    "val": 0,
    "test": 0,
}

for image_id in images:

    classes = {
        int(row["category_id"])
        for row in image_records[image_id]
    }

    available = [
        split
        for split in splits
        if split_image_counts[split]
        < target_counts[split]
    ]

    if not available:

        available = list(splits.keys())

    def score(split):

        rarity_load = sum(
            split_class_counts[split][c]
            for c in classes
        )

        image_load = split_image_counts[split]

        return (
            rarity_load,
            image_load,
            random.random(),
        )

    chosen = min(
        available,
        key=score
    )

    splits[chosen].append(image_id)

    split_image_counts[chosen] += 1

    for c in classes:

        split_class_counts[
            chosen
        ][c] += 1

print("\nImage-level split:")

for split in ["train", "val", "test"]:

    print(
        f"  {split:<5}: "
        f"{len(splits[split]):>4} images"
    )

# ------------------------------------------------------------
# Clean previous generated dataset
# ------------------------------------------------------------

if OUTPUT.exists():

    print(
        "\nRemoving previous generated dataset..."
    )

    shutil.rmtree(OUTPUT)

for split in ["train", "val", "test"]:

    (
        OUTPUT
        / "images"
        / split
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    (
        OUTPUT
        / "labels"
        / split
    ).mkdir(
        parents=True,
        exist_ok=True
    )

# ------------------------------------------------------------
# Convert annotations
# ------------------------------------------------------------

split_boxes = Counter()
class_boxes = Counter()
class_split_boxes = defaultdict(Counter)

copied_images = 0
invalid_boxes = 0
duplicate_destination_names = 0

used_destination_names = {
    "train": set(),
    "val": set(),
    "test": set(),
}

for split, split_images in splits.items():

    for image_id in split_images:

        record = image_manifest[image_id]

        source_image = record["image_path"]

        # Use the image_id in the generated filename.
        # This guarantees no filename collision.
        suffix = source_image.suffix.lower()

        if suffix not in {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp",
        }:

            print(
                f"WARNING: unsupported image format: "
                f"{source_image}"
            )

            continue

        output_stem = (
            f"taco_{image_id}"
        )

        destination_image = (
            OUTPUT
            / "images"
            / split
            / f"{output_stem}{suffix}"
        )

        destination_label = (
            OUTPUT
            / "labels"
            / split
            / f"{output_stem}.txt"
        )

        if output_stem in used_destination_names[split]:

            duplicate_destination_names += 1

        used_destination_names[
            split
        ].add(output_stem)

        # Verify actual image dimensions.

        with Image.open(source_image) as image:

            actual_width, actual_height = image.size

        expected_width = record["width"]
        expected_height = record["height"]

        if (
            actual_width != expected_width
            or actual_height != expected_height
        ):

            print(
                f"WARNING: dimension mismatch "
                f"for image_id={image_id}: "
                f"manifest={expected_width}x{expected_height}, "
                f"actual={actual_width}x{actual_height}"
            )

        width = actual_width
        height = actual_height

        shutil.copy2(
            source_image,
            destination_image
        )

        lines = []

        for row in image_records[image_id]:

            taco_category = int(
                row["category_id"]
            )

            yolo_class = TACO_TO_YOLO[
                taco_category
            ]

            x = float(row["x"])
            y = float(row["y"])
            w = float(row["width"])
            h = float(row["height"])

            # Clip to actual image boundaries.

            x1 = max(
                0.0,
                min(x, width)
            )

            y1 = max(
                0.0,
                min(y, height)
            )

            x2 = max(
                0.0,
                min(x + w, width)
            )

            y2 = max(
                0.0,
                min(y + h, height)
            )

            box_width = x2 - x1
            box_height = y2 - y1

            if (
                box_width <= 0
                or box_height <= 0
            ):

                invalid_boxes += 1
                continue

            center_x = (
                x1 + box_width / 2
            ) / width

            center_y = (
                y1 + box_height / 2
            ) / height

            norm_width = (
                box_width / width
            )

            norm_height = (
                box_height / height
            )

            if not (
                0 <= center_x <= 1
                and 0 <= center_y <= 1
                and 0 < norm_width <= 1
                and 0 < norm_height <= 1
            ):

                invalid_boxes += 1
                continue

            lines.append(
                f"{yolo_class} "
                f"{center_x:.6f} "
                f"{center_y:.6f} "
                f"{norm_width:.6f} "
                f"{norm_height:.6f}"
            )

            split_boxes[split] += 1
            class_boxes[taco_category] += 1
            class_split_boxes[
                taco_category
            ][split] += 1

        destination_label.write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8"
        )

        copied_images += 1

# ------------------------------------------------------------
# data.yaml
# ------------------------------------------------------------

yaml_data = {
    "path": str(OUTPUT.resolve()),
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "nc": len(NAMES),
    "names": NAMES,
}

with open(
    OUTPUT / "data.yaml",
    "w",
    encoding="utf-8"
) as f:

    yaml.safe_dump(
        yaml_data,
        f,
        sort_keys=False,
        allow_unicode=True
    )

# ------------------------------------------------------------
# Class mapping
# ------------------------------------------------------------

with open(
    OUTPUT / "class_mapping.csv",
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "yolo_id",
        "taco_id",
        "class_name",
    ])

    for yolo_id, (taco_id, name) in enumerate(
        SELECTED
    ):

        writer.writerow([
            yolo_id,
            taco_id,
            name,
        ])

# ------------------------------------------------------------
# Split manifest
# ------------------------------------------------------------

with open(
    OUTPUT / "split_manifest.csv",
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "image_id",
        "split",
        "source_path",
        "generated_filename",
    ])

    for split, split_images in splits.items():

        for image_id in split_images:

            source = image_manifest[
                image_id
            ]["image_path"]

            writer.writerow([
                image_id,
                split,
                str(source),
                f"taco_{image_id}",
            ])

# ------------------------------------------------------------
# Final verification
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("DATASET BUILD COMPLETE")
print("=" * 75)

print(
    f"\nImages copied:          {copied_images:,}"
)

print(
    f"Converted boxes:       "
    f"{sum(split_boxes.values()):,}"
)

print(
    f"Invalid boxes skipped: {invalid_boxes}"
)

print(
    f"Filename collisions:   "
    f"{duplicate_destination_names}"
)

print("\nSplit statistics:")

for split in ["train", "val", "test"]:

    image_count = len(splits[split])

    print(
        f"  {split:<5}: "
        f"{image_count:>4} images | "
        f"{split_boxes[split]:>5} boxes"
    )

print("\nClass distribution across splits:")

for yolo_id, (taco_id, name) in enumerate(
    SELECTED
):

    c = class_split_boxes[taco_id]

    print(
        f"{yolo_id:>2} | "
        f"{name:<32} | "
        f"train={c['train']:>3} "
        f"val={c['val']:>3} "
        f"test={c['test']:>3}"
    )

print(
    f"\nGenerated dataset:\n{OUTPUT}"
)

print(
    "\nOriginal TACO source remains untouched."
)

print(
    "\nNext step: VERIFY labels and visualize "
    "sample bounding boxes before training."
)
