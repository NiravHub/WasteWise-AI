from pathlib import Path
import csv
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]

AUDIT = ROOT / "results" / "dataset_audit" / "taco"

annotations_file = AUDIT / "taco_clean_annotations.csv"
categories_file = AUDIT / "taco_categories.csv"

counts = Counter()

with open(
    annotations_file,
    "r",
    encoding="utf-8"
) as f:
    for row in csv.DictReader(f):
        counts[int(row["category_id"])] += 1

categories = {}

with open(
    categories_file,
    "r",
    encoding="utf-8"
) as f:
    for row in csv.DictReader(f):
        categories[int(row["category_id"])] = row

print("=" * 70)
print("TACO DETECTION CLASS SELECTION ANALYSIS")
print("=" * 70)

rows = []

for category_id, category in categories.items():

    count = counts.get(category_id, 0)

    if count >= 100:
        tier = "STRONG"
    elif count >= 50:
        tier = "USABLE"
    elif count >= 20:
        tier = "LIMITED"
    elif count > 0:
        tier = "RARE"
    else:
        tier = "EMPTY"

    rows.append({
        "category_id": category_id,
        "name": category["name"],
        "supercategory": category["supercategory"],
        "annotations": count,
        "tier": tier,
    })

rows.sort(
    key=lambda x: x["annotations"],
    reverse=True
)

for row in rows:

    print(
        f"{row['category_id']:>2} | "
        f"{row['annotations']:>4} | "
        f"{row['tier']:<8} | "
        f"{row['name']}"
    )

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

for tier in [
    "STRONG",
    "USABLE",
    "LIMITED",
    "RARE",
    "EMPTY",
]:
    subset = [
        r for r in rows
        if r["tier"] == tier
    ]

    print(
        f"{tier:<10}: "
        f"{len(subset)} classes"
    )

print("\nRecommended initial detector candidates:")

for row in rows:

    if row["annotations"] >= 50:

        print(
            f"  {row['name']:<35} "
            f"{row['annotations']} annotations"
        )

print("\nNOTE:")
print(
    "This is a statistical candidate list only. "
    "We will make the final taxonomy decision after "
    "reviewing the class names and visual overlap with "
    "the existing WasteWise taxonomy."
)
