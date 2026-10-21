from pathlib import Path
import json
import csv
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]

TACO_ROOT = (
    ROOT / "data" / "external" / "taco" /
    "extracted" / "TACO"
)

DATA_ROOT = TACO_ROOT / "data"

OUT_DIR = ROOT / "results" / "dataset_audit" / "taco"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("TACO CLEAN ANNOTATION PREPARATION")
print("=" * 70)

json_files = sorted(DATA_ROOT.rglob("annotations.json"))

# ------------------------------------------------------------
# Load categories
# ------------------------------------------------------------

categories = {}

# ------------------------------------------------------------
# Deduplicate images
# ------------------------------------------------------------

images = {}

# ------------------------------------------------------------
# Deduplicate annotations
# ------------------------------------------------------------

annotations = {}

for json_path in json_files:

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for category in data.get("categories", []):
        categories[category["id"]] = category

    for image in data.get("images", []):

        image_id = image["id"]

        if image_id not in images:
            images[image_id] = image

    for annotation in data.get("annotations", []):

        ann_id = annotation["id"]

        # Same annotation ID appearing in multiple
        # annotation files is treated as the same annotation.
        if ann_id not in annotations:
            annotations[ann_id] = annotation


print(f"\nAnnotation files: {len(json_files)}")
print(f"Unique images:    {len(images)}")
print(f"Unique annotations: {len(annotations)}")
print(f"Categories:       {len(categories)}")


# ------------------------------------------------------------
# Resolve image files
# ------------------------------------------------------------

image_files = {}

for p in DATA_ROOT.rglob("*"):

    if not p.is_file():
        continue

    if p.suffix.lower() not in {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    }:
        continue

    image_files[p.name] = p


# TACO image records normally contain file_name.
# Build additional lookup by basename.
resolved_images = {}
unresolved_images = []

for image_id, image in images.items():

    file_name = image.get("file_name", "")

    basename = Path(file_name).name

    path = image_files.get(basename)

    if path is None:

        # fallback: search by exact relative path
        candidate = DATA_ROOT / file_name

        if candidate.exists():
            path = candidate

    if path is None:

        unresolved_images.append({
            "image_id": image_id,
            "file_name": file_name,
        })

    else:

        resolved_images[image_id] = path


print(f"\nResolved image paths:   {len(resolved_images)}")
print(f"Unresolved image paths: {len(unresolved_images)}")


# ------------------------------------------------------------
# Process annotations
# ------------------------------------------------------------

clean_annotations = []

stats = Counter()

category_stats = defaultdict(Counter)

for ann_id, ann in annotations.items():

    image_id = ann["image_id"]

    if image_id not in images:
        stats["missing_image"] += 1
        continue

    image = images[image_id]

    W = float(image["width"])
    H = float(image["height"])

    bbox = ann.get("bbox")

    if not isinstance(bbox, list) or len(bbox) != 4:
        stats["invalid_bbox"] += 1
        continue

    x, y, w, h = map(float, bbox)

    original_x1 = x
    original_y1 = y
    original_x2 = x + w
    original_y2 = y + h

    # Clip to image boundaries.
    x1 = max(0.0, min(W, original_x1))
    y1 = max(0.0, min(H, original_y1))
    x2 = max(0.0, min(W, original_x2))
    y2 = max(0.0, min(H, original_y2))

    repaired = (
        x1 != original_x1
        or y1 != original_y1
        or x2 != original_x2
        or y2 != original_y2
    )

    if repaired:
        stats["repaired"] += 1

    new_w = x2 - x1
    new_h = y2 - y1

    if new_w <= 0 or new_h <= 0:
        stats["removed_zero_area_after_clipping"] += 1
        category_stats[
            ann["category_id"]
        ]["removed"] += 1
        continue

    if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
        stats["removed_invalid_after_clipping"] += 1
        continue

    category_stats[
        ann["category_id"]
    ]["kept"] += 1

    clean_annotations.append({
        "annotation_id": ann_id,
        "image_id": image_id,
        "category_id": ann["category_id"],
        "x": x1,
        "y": y1,
        "width": new_w,
        "height": new_h,
        "area": new_w * new_h,
        "iscrowd": ann.get("iscrowd", 0),
        "repaired": repaired,
    })


# ------------------------------------------------------------
# Save clean annotation manifest
# ------------------------------------------------------------

manifest_path = OUT_DIR / "taco_clean_annotations.csv"

with open(
    manifest_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "annotation_id",
            "image_id",
            "category_id",
            "x",
            "y",
            "width",
            "height",
            "area",
            "iscrowd",
            "repaired",
        ],
    )

    writer.writeheader()
    writer.writerows(clean_annotations)


# ------------------------------------------------------------
# Save image manifest
# ------------------------------------------------------------

image_manifest_path = OUT_DIR / "taco_image_manifest.csv"

with open(
    image_manifest_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "image_id",
            "file_name",
            "width",
            "height",
            "image_path",
        ],
    )

    writer.writeheader()

    for image_id, image in sorted(images.items()):

        writer.writerow({
            "image_id": image_id,
            "file_name": image.get("file_name", ""),
            "width": image.get("width", ""),
            "height": image.get("height", ""),
            "image_path": str(
                resolved_images[image_id]
            ) if image_id in resolved_images else "",
        })


# ------------------------------------------------------------
# Save category manifest
# ------------------------------------------------------------

category_manifest_path = OUT_DIR / "taco_categories.csv"

with open(
    category_manifest_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "category_id",
            "name",
            "supercategory",
        ],
    )

    writer.writeheader()

    for category_id in sorted(categories):

        category = categories[category_id]

        writer.writerow({
            "category_id": category_id,
            "name": category.get("name", ""),
            "supercategory": category.get(
                "supercategory", ""
            ),
        })


# ------------------------------------------------------------
# Save summary
# ------------------------------------------------------------

summary = {
    "annotation_files": len(json_files),
    "unique_images": len(images),
    "unique_annotations": len(annotations),
    "resolved_images": len(resolved_images),
    "unresolved_images": len(unresolved_images),
    "clean_annotations": len(clean_annotations),
    "statistics": dict(stats),
}

summary_path = OUT_DIR / "taco_cleaning_summary.json"

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:
    json.dump(summary, f, indent=2)


# ------------------------------------------------------------
# Print
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CLEANING RESULTS")
print("=" * 70)

print(f"Unique images:                 {len(images)}")
print(f"Unique annotations:            {len(annotations)}")
print(f"Resolved image files:          {len(resolved_images)}")
print(f"Unresolved image files:        {len(unresolved_images)}")
print(f"Clean annotations:             {len(clean_annotations)}")
print(f"Repaired/clipped boxes:        {stats['repaired']}")
print(
    "Removed after clipping:        "
    f"{stats['removed_zero_area_after_clipping']}"
)
print(f"Invalid boxes:                 {stats['invalid_bbox']}")

print("\nSaved:")
print(manifest_path)
print(image_manifest_path)
print(category_manifest_path)
print(summary_path)

print("\nTACO CLEANING PREPARATION COMPLETE")
