from pathlib import Path
import json
import csv
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]
TACO_ROOT = ROOT / "data" / "external" / "taco" / "extracted" / "TACO"
DATA_ROOT = TACO_ROOT / "data"

OUT_DIR = ROOT / "results" / "dataset_audit" / "taco"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("WasteWise AI - TACO Detection Dataset Audit")
print("=" * 70)

json_files = sorted(DATA_ROOT.rglob("annotations.json"))

print(f"\nAnnotation files found: {len(json_files)}")

all_categories = {}
all_images = []
all_annotations = []

image_sources = defaultdict(list)

for json_path in json_files:
    print(f"\nReading: {json_path.relative_to(TACO_ROOT)}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    categories = data.get("categories", [])
    images = data.get("images", [])
    annotations = data.get("annotations", [])

    print(f"  Images:      {len(images)}")
    print(f"  Annotations: {len(annotations)}")
    print(f"  Categories:  {len(categories)}")

    for cat in categories:
        all_categories[cat["id"]] = cat

    for img in images:
        record = dict(img)
        record["_annotation_file"] = str(json_path.relative_to(TACO_ROOT))
        all_images.append(record)
        image_sources[img["id"]].append(str(json_path.relative_to(TACO_ROOT)))

    for ann in annotations:
        record = dict(ann)
        record["_annotation_file"] = str(json_path.relative_to(TACO_ROOT))
        all_annotations.append(record)


# ------------------------------------------------------------
# Basic statistics
# ------------------------------------------------------------

category_counts = Counter()

for ann in all_annotations:
    category_counts[ann["category_id"]] += 1


image_annotation_counts = Counter()

for ann in all_annotations:
    image_annotation_counts[ann["image_id"]] += 1


# ------------------------------------------------------------
# Image integrity
# ------------------------------------------------------------

image_dir_candidates = [
    DATA_ROOT,
    TACO_ROOT,
]

image_files = []

for base in image_dir_candidates:
    if base.exists():
        image_files.extend(
            p for p in base.rglob("*")
            if p.is_file()
            and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )

image_files = sorted(set(image_files))

print("\n" + "=" * 70)
print("IMAGE INVENTORY")
print("=" * 70)

print(f"Recognized image files: {len(image_files)}")
print(f"Annotation image records: {len(all_images)}")

# ------------------------------------------------------------
# Annotation integrity
# ------------------------------------------------------------

image_ids = {img["id"] for img in all_images}
annotation_image_ids = {ann["image_id"] for ann in all_annotations}
category_ids = set(all_categories.keys())

missing_image_refs = sorted(annotation_image_ids - image_ids)
unknown_category_refs = sorted(
    {ann["category_id"] for ann in all_annotations} - category_ids
)

invalid_bbox = []
zero_area_bbox = []
out_of_bounds_bbox = []

image_dimensions = {
    img["id"]: (
        float(img.get("width", 0)),
        float(img.get("height", 0))
    )
    for img in all_images
}

for ann in all_annotations:
    bbox = ann.get("bbox")

    if not isinstance(bbox, list) or len(bbox) != 4:
        invalid_bbox.append(ann["id"])
        continue

    x, y, w, h = map(float, bbox)

    if w <= 0 or h <= 0:
        zero_area_bbox.append(ann["id"])

    width, height = image_dimensions.get(
        ann["image_id"],
        (0, 0)
    )

    if width > 0 and height > 0:
        if x < 0 or y < 0 or x + w > width or y + h > height:
            out_of_bounds_bbox.append(ann["id"])


# ------------------------------------------------------------
# Category table
# ------------------------------------------------------------

category_rows = []

for category_id in sorted(all_categories):
    cat = all_categories[category_id]

    category_rows.append({
        "category_id": category_id,
        "name": cat.get("name", ""),
        "supercategory": cat.get("supercategory", ""),
        "annotation_count": category_counts.get(category_id, 0),
    })


category_rows.sort(
    key=lambda x: x["annotation_count"],
    reverse=True
)

category_csv = OUT_DIR / "taco_category_distribution.csv"

with open(category_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "category_id",
            "name",
            "supercategory",
            "annotation_count",
        ],
    )
    writer.writeheader()
    writer.writerows(category_rows)


# ------------------------------------------------------------
# Image annotation distribution
# ------------------------------------------------------------

annotated_images = len(annotation_image_ids)
unannotated_images = len(image_ids - annotation_image_ids)

annotation_distribution = Counter(
    image_annotation_counts.values()
)

# ------------------------------------------------------------
# Bounding-box statistics
# ------------------------------------------------------------

bbox_widths = []
bbox_heights = []
bbox_areas = []
bbox_area_ratios = []

for ann in all_annotations:
    bbox = ann.get("bbox")

    if not isinstance(bbox, list) or len(bbox) != 4:
        continue

    x, y, w, h = map(float, bbox)

    bbox_widths.append(w)
    bbox_heights.append(h)
    bbox_areas.append(w * h)

    width, height = image_dimensions.get(
        ann["image_id"],
        (0, 0)
    )

    if width > 0 and height > 0:
        bbox_area_ratios.append(
            (w * h) / (width * height)
        )


def stats(values):
    if not values:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "median": None,
            "max": None,
        }

    values = sorted(values)
    n = len(values)

    if n % 2:
        median = values[n // 2]
    else:
        median = (
            values[n // 2 - 1] +
            values[n // 2]
        ) / 2

    return {
        "count": len(values),
        "min": min(values),
        "mean": sum(values) / len(values),
        "median": median,
        "max": max(values),
    }


# ------------------------------------------------------------
# Save complete audit JSON
# ------------------------------------------------------------

audit = {
    "dataset": "TACO",
    "annotation_files": len(json_files),
    "annotation_file_paths": [
        str(p.relative_to(TACO_ROOT))
        for p in json_files
    ],
    "image_files_found": len(image_files),
    "annotation_image_records": len(all_images),
    "unique_image_ids": len(image_ids),
    "annotated_image_ids": annotated_images,
    "unannotated_image_ids": unannotated_images,
    "total_annotations": len(all_annotations),
    "unique_categories": len(all_categories),
    "missing_annotation_image_refs": len(missing_image_refs),
    "unknown_category_refs": len(unknown_category_refs),
    "invalid_bbox": len(invalid_bbox),
    "zero_area_bbox": len(zero_area_bbox),
    "out_of_bounds_bbox": len(out_of_bounds_bbox),
    "images_with_annotation_count_distribution": dict(
        sorted(annotation_distribution.items())
    ),
    "bbox_width": stats(bbox_widths),
    "bbox_height": stats(bbox_heights),
    "bbox_area": stats(bbox_areas),
    "bbox_area_ratio": stats(bbox_area_ratios),
    "categories": category_rows,
}

audit_json = OUT_DIR / "taco_audit.json"

with open(audit_json, "w", encoding="utf-8") as f:
    json.dump(audit, f, indent=2)


# ------------------------------------------------------------
# Print results
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TACO AUDIT RESULTS")
print("=" * 70)

print(f"Annotation files:              {len(json_files)}")
print(f"Image files found:             {len(image_files)}")
print(f"Annotation image records:      {len(all_images)}")
print(f"Unique image IDs:              {len(image_ids)}")
print(f"Annotated image IDs:           {annotated_images}")
print(f"Unannotated image IDs:         {unannotated_images}")
print(f"Total annotations:             {len(all_annotations)}")
print(f"Unique categories:             {len(all_categories)}")

print("\nAnnotation integrity:")
print(f"  Missing image references:    {len(missing_image_refs)}")
print(f"  Unknown category references: {len(unknown_category_refs)}")
print(f"  Invalid bounding boxes:      {len(invalid_bbox)}")
print(f"  Zero-area bounding boxes:    {len(zero_area_bbox)}")
print(f"  Out-of-bounds bounding boxes:{len(out_of_bounds_bbox)}")

print("\nBounding-box statistics:")

for name, values in [
    ("Width", bbox_widths),
    ("Height", bbox_heights),
    ("Area", bbox_areas),
    ("Area ratio", bbox_area_ratios),
]:
    s = stats(values)
    print(
        f"  {name:<12} "
        f"min={s['min']:.4f} "
        f"mean={s['mean']:.4f} "
        f"median={s['median']:.4f} "
        f"max={s['max']:.4f}"
    )

print("\nTop categories by annotation count:")

for row in category_rows[:15]:
    print(
        f"  {row['category_id']:>2}  "
        f"{row['name']:<30} "
        f"{row['annotation_count']:>6}"
    )

print("\nBottom categories by annotation count:")

for row in category_rows[-15:]:
    print(
        f"  {row['category_id']:>2}  "
        f"{row['name']:<30} "
        f"{row['annotation_count']:>6}"
    )

print("\nSaved:")
print(audit_json)
print(category_csv)

print("\nAUDIT COMPLETE")
