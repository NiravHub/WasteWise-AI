import json
import random
import shutil
from pathlib import Path

from PIL import Image, ImageOps


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TACO_ROOT = ROOT / "data" / "external" / "taco" / "extracted" / "TACO"
ANNOTATION_FILE = TACO_ROOT / "data" / "annotations.json"

SOURCE_IMAGE_ROOT = TACO_ROOT / "data"

OUTPUT_ROOT = ROOT / "data" / "processed" / "taco_detection_complete"

SEED = 42

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10


# ============================================================
# HELPERS
# ============================================================

def yolo_value(value):
    return f"{value:.6f}"


def convert_bbox(x, y, w, h, image_width, image_height):
    """
    TACO bounding boxes are already expressed in the
    display-oriented coordinate system.

    Images are normalized using ImageOps.exif_transpose().
    Therefore, DO NOT rotate/transform the bbox.

    Only clip tiny source-boundary violations.
    """

    left = max(0.0, float(x))
    top = max(0.0, float(y))
    right = min(float(image_width), float(x) + float(w))
    bottom = min(float(image_height), float(y) + float(h))

    clipped_w = right - left
    clipped_h = bottom - top

    if clipped_w <= 0 or clipped_h <= 0:
        return None

    cx = (left + right) / 2.0
    cy = (top + bottom) / 2.0

    x_center = cx / image_width
    y_center = cy / image_height
    width = clipped_w / image_width
    height = clipped_h / image_height

    if not (
        0.0 <= x_center <= 1.0
        and 0.0 <= y_center <= 1.0
        and 0.0 < width <= 1.0
        and 0.0 < height <= 1.0
    ):
        return None

    return x_center, y_center, width, height


# ============================================================
# LOAD TACO MASTER JSON
# ============================================================

print("=" * 70)
print("BUILD COMPLETE TACO 59-CLASS DETECTION DATASET")
print("=" * 70)

with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
    taco = json.load(f)

images = taco["images"]
annotations = taco["annotations"]
categories = taco["categories"]

print(f"Master images:       {len(images)}")
print(f"Master annotations:  {len(annotations)}")
print(f"Category definitions: {len(categories)}")


# ============================================================
# ACTIVE CATEGORIES
# ============================================================

annotation_counts = {}

for ann in annotations:
    cid = ann["category_id"]
    annotation_counts[cid] = annotation_counts.get(cid, 0) + 1

active_categories = [
    cat for cat in categories
    if annotation_counts.get(cat["id"], 0) > 0
]

active_categories.sort(key=lambda x: x["id"])

category_id_to_yolo = {
    cat["id"]: idx
    for idx, cat in enumerate(active_categories)
}

category_id_to_name = {
    cat["id"]: cat["name"]
    for cat in active_categories
}

print(f"Active categories:  {len(active_categories)}")

assert len(active_categories) == 59
assert len(annotations) == 4784


# ============================================================
# PRINT CLASS MAP
# ============================================================

print("\nCLASS MAP")
print("-" * 70)

for cat in active_categories:
    cid = cat["id"]
    print(
        f"{category_id_to_yolo[cid]:2d} | "
        f"TACO ID {cid:2d} | "
        f"{cat['name']} | "
        f"{annotation_counts[cid]} annotations"
    )


# ============================================================
# GROUP ANNOTATIONS BY IMAGE
# ============================================================

annotations_by_image = {}

for ann in annotations:
    annotations_by_image.setdefault(
        ann["image_id"], []
    ).append(ann)


# ============================================================
# RESOLVE SOURCE IMAGE PATH
# ============================================================

def resolve_image_path(file_name):
    """
    Master TACO annotations use paths such as:

        batch_1/000006.jpg

    Resolve relative to TACO/data/.
    """

    path = SOURCE_IMAGE_ROOT / file_name

    if path.exists():
        return path

    # Case-insensitive fallback
    parts = Path(file_name).parts

    if len(parts) >= 2:
        batch_dir = SOURCE_IMAGE_ROOT / parts[0]
        target_name = parts[-1].lower()

        if batch_dir.exists():
            for candidate in batch_dir.iterdir():
                if candidate.name.lower() == target_name:
                    return candidate

    return None


# ============================================================
# PREPARE OUTPUT
# ============================================================

if OUTPUT_ROOT.exists():
    print("\nRemoving previous incomplete complete-dataset build...")
    shutil.rmtree(OUTPUT_ROOT)

for split in ["train", "val", "test"]:
    (OUTPUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)


# ============================================================
# IMAGE-LEVEL SPLIT
# ============================================================

rng = random.Random(SEED)

image_records = []

for image in images:
    image_id = image["id"]

    anns = annotations_by_image.get(image_id, [])

    if not anns:
        continue

    image_records.append(image)

rng.shuffle(image_records)

total_images = len(image_records)

train_end = int(total_images * TRAIN_RATIO)
val_end = train_end + int(total_images * VAL_RATIO)

split_images = {
    "train": image_records[:train_end],
    "val": image_records[train_end:val_end],
    "test": image_records[val_end:],
}

print("\nSPLIT")
print("-" * 70)
print(f"Annotated images: {total_images}")
print(f"Train:            {len(split_images['train'])}")
print(f"Val:              {len(split_images['val'])}")
print(f"Test:             {len(split_images['test'])}")


# ============================================================
# PROCESS IMAGES + LABELS
# ============================================================

stats = {
    "selected_images": total_images,
    "processed_images": 0,
    "selected_annotations": len(annotations),
    "converted_boxes": 0,
    "skipped_boxes": 0,
    "invalid_boxes": 0,
    "missing_images": 0,
    "dimension_mismatches": 0,
    "orientation_counts": {},
}


for split, split_records in split_images.items():

    print(f"\nProcessing {split}...")

    for index, image_record in enumerate(split_records, start=1):

        image_id = image_record["id"]
        file_name = image_record["file_name"]

        source_path = resolve_image_path(file_name)

        if source_path is None:
            stats["missing_images"] += 1
            print(f"[MISSING] {file_name}")
            continue

        try:
            with Image.open(source_path) as original:

                orientation = original.getexif().get(274, 1)

                stats["orientation_counts"][orientation] = (
                    stats["orientation_counts"].get(orientation, 0) + 1
                )

                # CRITICAL:
                # Normalize EXIF orientation.
                normalized = ImageOps.exif_transpose(original)

                width, height = normalized.size

                expected_width = image_record.get("width")
                expected_height = image_record.get("height")

                if (
                    expected_width is not None
                    and expected_height is not None
                    and (
                        width != expected_width
                        or height != expected_height
                    )
                ):
                    # This is not necessarily an error.
                    # TACO metadata can describe the pre-display orientation.
                    stats["dimension_mismatches"] += 1

                # Convert to RGB.
                if normalized.mode != "RGB":
                    normalized = normalized.convert("RGB")

                output_stem = f"taco_{image_id:04d}"

                output_image = (
                    OUTPUT_ROOT
                    / "images"
                    / split
                    / f"{output_stem}.jpg"
                )

                output_label = (
                    OUTPUT_ROOT
                    / "labels"
                    / split
                    / f"{output_stem}.txt"
                )

                normalized.save(
                    output_image,
                    format="JPEG",
                    quality=95
                )

        except Exception as exc:
            stats["missing_images"] += 1
            print(f"[ERROR] {file_name}: {exc}")
            continue

        label_lines = []

        for ann in annotations_by_image[image_id]:

            category_id = ann["category_id"]

            if category_id not in category_id_to_yolo:
                stats["skipped_boxes"] += 1
                continue

            bbox = ann["bbox"]

            if len(bbox) != 4:
                stats["invalid_boxes"] += 1
                continue

            x, y, w, h = bbox

            converted = convert_bbox(
                x,
                y,
                w,
                h,
                width,
                height,
            )

            if converted is None:
                stats["invalid_boxes"] += 1
                continue

            xc, yc, nw, nh = converted

            class_id = category_id_to_yolo[category_id]

            label_lines.append(
                f"{class_id} "
                f"{yolo_value(xc)} "
                f"{yolo_value(yc)} "
                f"{yolo_value(nw)} "
                f"{yolo_value(nh)}"
            )

            stats["converted_boxes"] += 1

        if not label_lines:
            # Remove image if no usable annotations survived.
            output_image.unlink(missing_ok=True)
            stats["invalid_boxes"] += 1
            continue

        output_label.write_text(
            "\n".join(label_lines) + "\n",
            encoding="utf-8"
        )

        stats["processed_images"] += 1

        if index % 100 == 0 or index == len(split_records):
            print(
                f"  {index}/{len(split_records)}"
            )


# ============================================================
# DATASET YAML
# ============================================================

yaml_lines = [
    "path: .",
    "train: images/train",
    "val: images/val",
    "test: images/test",
    "",
    "names:",
]

for index, category in enumerate(active_categories):
    # Quote names safely.
    name = category["name"].replace('"', '\\"')
    yaml_lines.append(
        f'  {index}: "{name}"'
    )

(OUTPUT_ROOT / "data.yaml").write_text(
    "\n".join(yaml_lines) + "\n",
    encoding="utf-8"
)


# ============================================================
# CLASS MANIFEST
# ============================================================

manifest_lines = [
    "yolo_id,taco_id,class_name,annotation_count"
]

for category in active_categories:
    cid = category["id"]

    manifest_lines.append(
        f"{category_id_to_yolo[cid]},"
        f"{cid},"
        f"\"{category['name'].replace(chr(34), chr(34)*2)}\","
        f"{annotation_counts[cid]}"
    )

(OUTPUT_ROOT / "class_manifest.csv").write_text(
    "\n".join(manifest_lines) + "\n",
    encoding="utf-8"
)


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("BUILD COMPLETE")
print("=" * 70)

print(f"Selected images:       {stats['selected_images']}")
print(f"Processed images:      {stats['processed_images']}")
print(f"Selected annotations:  {stats['selected_annotations']}")
print(f"Converted boxes:       {stats['converted_boxes']}")
print(f"Skipped boxes:         {stats['skipped_boxes']}")
print(f"Invalid boxes:         {stats['invalid_boxes']}")
print(f"Missing images:        {stats['missing_images']}")
print(f"Dimension mismatches:  {stats['dimension_mismatches']}")

print("\nEXIF orientations:")
for orientation, count in sorted(
    stats["orientation_counts"].items()
):
    print(f"  {orientation}: {count}")

print("\nOutput:")
print(OUTPUT_ROOT)

print("\nExpected:")
print("  59 active classes")
print("  4784 source annotations")
print("  0 skipped annotations")
print("  0 invalid boxes")
print("=" * 70)