import json
from pathlib import Path
from collections import Counter

root = Path.cwd()
ann_path = root / "data/external/taco/extracted/TACO/data/annotations.json"

with open(ann_path, "r", encoding="utf-8") as f:
    data = json.load(f)

counts = Counter(a["category_id"] for a in data["annotations"])

active_ids = {
    cid for cid, count in counts.items()
    if count > 0
}

selected = [
    a for a in data["annotations"]
    if a["category_id"] in active_ids
]

print("=" * 70)
print("COMPLETE TACO ANNOTATION COUNT")
print("=" * 70)
print(f"Total annotations in JSON: {len(data['annotations'])}")
print(f"Active categories:          {len(active_ids)}")
print(f"Annotations in 59 classes:  {len(selected)}")
print(f"Zero-instance categories:   {60 - len(active_ids)}")
