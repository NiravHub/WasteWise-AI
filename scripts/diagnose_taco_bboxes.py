from pathlib import Path
import json
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]
TACO = ROOT / "data" / "external" / "taco" / "extracted" / "TACO"
DATA = TACO / "data"

json_files = sorted(DATA.rglob("annotations.json"))

images = {}
annotations = []

for path in json_files:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for img in data.get("images", []):
        images[img["id"]] = img

    annotations.extend(data.get("annotations", []))

print("=" * 70)
print("TACO OUT-OF-BOUNDS BBOX DIAGNOSTIC")
print("=" * 70)

bad = []

for ann in annotations:
    img = images[ann["image_id"]]

    x, y, w, h = map(float, ann["bbox"])
    W = float(img["width"])
    H = float(img["height"])

    if x < 0 or y < 0 or x + w > W or y + h > H:
        bad.append({
            "ann_id": ann["id"],
            "image_id": ann["image_id"],
            "category_id": ann["category_id"],
            "bbox": ann["bbox"],
            "image_width": W,
            "image_height": H,
            "right": x + w,
            "bottom": y + h,
            "source": ann.get("_annotation_file", "")
        })

print(f"\nTotal annotations: {len(annotations)}")
print(f"Out-of-bounds: {len(bad)}")

print("\nExamples:")
for item in bad[:30]:
    print(
        f"ann={item['ann_id']} "
        f"image={item['image_id']} "
        f"category={item['category_id']} "
        f"bbox={item['bbox']} "
        f"image={int(item['image_width'])}x{int(item['image_height'])} "
        f"right={item['right']:.1f} "
        f"bottom={item['bottom']:.1f}"
    )

# How far outside?
right_excess = []
bottom_excess = []
left_excess = []
top_excess = []

for item in bad:
    x, y, w, h = map(float, item["bbox"])
    W = item["image_width"]
    H = item["image_height"]

    if x < 0:
        left_excess.append(abs(x))

    if y < 0:
        top_excess.append(abs(y))

    if x + w > W:
        right_excess.append(x + w - W)

    if y + h > H:
        bottom_excess.append(y + h - H)

print("\nViolation directions:")
print(f"  Left:   {len(left_excess)}")
print(f"  Top:    {len(top_excess)}")
print(f"  Right:  {len(right_excess)}")
print(f"  Bottom: {len(bottom_excess)}")

print("\nMaximum overflow:")
print(f"  Left:   {max(left_excess, default=0):.2f}")
print(f"  Top:    {max(top_excess, default=0):.2f}")
print(f"  Right:  {max(right_excess, default=0):.2f}")
print(f"  Bottom: {max(bottom_excess, default=0):.2f}")

# Category impact
category_bad = Counter(x["category_id"] for x in bad)
category_total = Counter(x["category_id"] for x in annotations)

print("\nCategories most affected:")
for cid, count in category_bad.most_common(20):
    print(
        f"  category {cid}: "
        f"{count}/{category_total[cid]} "
        f"({100*count/category_total[cid]:.1f}%)"
    )

print("\nDIAGNOSTIC COMPLETE")
