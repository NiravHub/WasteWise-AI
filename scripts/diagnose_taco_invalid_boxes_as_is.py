from pathlib import Path
import csv
import json
from collections import Counter, defaultdict
from PIL import Image, ImageOps

ROOT = Path.cwd()
MANIFEST = ROOT / "results/dataset_audit/taco/taco_image_manifest_corrected.csv"
OUT = ROOT / "results/dataset_audit/taco/invalid_bbox_diagnostic_as_is.csv"

SELECTED_CLASSES = {
    "Cigarette",
    "Unlabeled litter",
    "Plastic film",
    "Clear plastic bottle",
    "Other plastic",
    "Other plastic wrapper",
    "Drink can",
    "Plastic bottle cap",
    "Plastic straw",
    "Broken glass",
    "Styrofoam piece",
    "Glass bottle",
    "Disposable plastic cup",
    "Pop tab",
    "Other carton",
    "Normal paper",
    "Metal bottle cap",
    "Plastic lid",
    "Paper cup",
    "Corrugated carton",
    "Aluminium foil",
    "Single-use carrier bag",
    "Other plastic bottle",
}

TACO_ROOT = ROOT / "data/external/taco/extracted/TACO"

# Load master annotations
with open(TACO_ROOT / "data/annotations.json", "r", encoding="utf-8") as f:
    data = json.load(f)

images = {x["id"]: x for x in data["images"]}
categories = {x["id"]: x["name"] for x in data["categories"]}

anns_by_image = defaultdict(list)

for ann in data["annotations"]:
    cls = categories.get(ann["category_id"])
    if cls in SELECTED_CLASSES:
        anns_by_image[ann["image_id"]].append(ann)

rows = []
invalid = []

for image_id, anns in anns_by_image.items():

    meta = images[image_id]
    rel = meta["file_name"]
    source = TACO_ROOT / "data" / rel

    if not source.exists():
        print(f"MISSING: {rel}")
        continue

    try:
        with Image.open(source) as im:
            orientation = im.getexif().get(274, 1) or 1
            normalized = ImageOps.exif_transpose(im)
            W, H = normalized.size
    except Exception as e:
        print(f"ERROR: {rel} -> {e}")
        continue

    expected_W = int(meta["width"])
    expected_H = int(meta["height"])

    if (W, H) != (expected_W, expected_H):
        print(
            f"DIMENSION MISMATCH: {rel} "
            f"metadata={expected_W}x{expected_H} "
            f"normalized={W}x{H}"
        )

    for ann in anns:
        x, y, bw, bh = [float(v) for v in ann["bbox"]]

        reasons = []

        if x < 0:
            reasons.append("negative_x")
        if y < 0:
            reasons.append("negative_y")
        if x + bw > W:
            reasons.append("right_overflow")
        if y + bh > H:
            reasons.append("bottom_overflow")
        if bw <= 0:
            reasons.append("nonpositive_width")
        if bh <= 0:
            reasons.append("nonpositive_height")

        row = {
            "image_id": image_id,
            "file_name": rel,
            "orientation": orientation,
            "width": W,
            "height": H,
            "category": categories[ann["category_id"]],
            "x": x,
            "y": y,
            "bbox_width": bw,
            "bbox_height": bh,
            "right": x + bw,
            "bottom": y + bh,
            "valid": len(reasons) == 0,
            "reasons": "|".join(reasons),
        }

        rows.append(row)

        if reasons:
            invalid.append(row)

with open(OUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("=" * 75)
print("TACO AS-IS BBOX DIAGNOSTIC")
print("=" * 75)

print(f"\nSelected annotations: {len(rows)}")
print(f"Invalid annotations:  {len(invalid)}")

print("\nINVALIDITY REASONS")
print("-" * 50)

reason_counts = Counter()

for r in invalid:
    for reason in r["reasons"].split("|"):
        reason_counts[reason] += 1

for reason, count in reason_counts.most_common():
    print(f"{reason:<30}{count:>8}")

print("\nINVALID BOXES BY EXIF ORIENTATION")
print("-" * 50)

orientation_counts = Counter(r["orientation"] for r in invalid)

for orientation, count in sorted(orientation_counts.items()):
    print(f"orientation {orientation:<16}{count:>8}")

print("\nINVALID BOXES BY CLASS")
print("-" * 50)

class_counts = Counter(r["category"] for r in invalid)

for cls, count in class_counts.most_common():
    print(f"{cls:<30}{count:>8}")

print("\nTOP AFFECTED IMAGES")
print("-" * 50)

image_counts = Counter(r["file_name"] for r in invalid)

for name, count in image_counts.most_common(20):
    print(f"{count:>3} invalid | {name}")

print(f"\nDetailed CSV:")
print(OUT)
