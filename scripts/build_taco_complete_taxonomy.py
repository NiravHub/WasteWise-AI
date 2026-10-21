import json
from pathlib import Path
from collections import Counter

ROOT = Path.cwd()
ANN = ROOT / "data/external/taco/extracted/TACO/data/annotations.json"
OUT = ROOT / "results/dataset_audit/taco/taco_complete_taxonomy.csv"

with open(ANN, "r", encoding="utf-8") as f:
    data = json.load(f)

categories = {c["id"]: c["name"] for c in data["categories"]}

counts = Counter(
    ann["category_id"]
    for ann in data["annotations"]
)

rows = []

for taco_id, name in sorted(categories.items()):
    rows.append({
        "taco_category_id": taco_id,
        "category_name": name,
        "annotation_count": counts[taco_id],
        "active": counts[taco_id] > 0,
    })

import csv

with open(OUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "taco_category_id",
            "category_name",
            "annotation_count",
            "active",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print("=" * 75)
print("COMPLETE TACO TAXONOMY")
print("=" * 75)

for row in rows:
    print(
        f"{row['taco_category_id']:>2} | "
        f"{row['annotation_count']:>5} | "
        f"{'ACTIVE' if row['active'] else 'ZERO'} | "
        f"{row['category_name']}"
    )

print()
print(f"Total category definitions: {len(rows)}")
print(f"Active categories: {sum(r['active'] for r in rows)}")
print(f"Zero-instance categories: {sum(not r['active'] for r in rows)}")
print()
print(f"Saved: {OUT}")
