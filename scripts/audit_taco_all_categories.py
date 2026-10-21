import json
from pathlib import Path
from collections import Counter

ROOT = Path.cwd()
ANN = ROOT / "data/external/taco/extracted/TACO/data/annotations.json"

with open(ANN, "r", encoding="utf-8") as f:
    data = json.load(f)

categories = {c["id"]: c["name"] for c in data["categories"]}

counts = Counter(
    ann["category_id"]
    for ann in data["annotations"]
)

print("=" * 80)
print("FULL TACO CATEGORY AUDIT")
print("=" * 80)

for cat_id, name in sorted(
    categories.items(),
    key=lambda x: (-counts[x[0]], x[0])
):
    print(
        f"{cat_id:>3} | "
        f"{counts[cat_id]:>5} annotations | "
        f"{name}"
    )

print()
print("=" * 80)
print("CURRENTLY EXCLUDED BUT POTENTIALLY IMPORTANT")
print("=" * 80)

current_selected = {
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

for cat_id, name in sorted(categories.items()):
    if name not in current_selected:
        print(
            f"{cat_id:>3} | "
            f"{counts[cat_id]:>5} annotations | "
            f"{name}"
        )
