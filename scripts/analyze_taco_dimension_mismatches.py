import csv
from collections import Counter
from pathlib import Path
from PIL import Image

manifest = Path("results/dataset_audit/taco/taco_image_manifest_corrected.csv")

mismatches = []

with open(manifest, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["status"] == "DIMENSION_MISMATCH":
            with Image.open(row["image_path"]) as im:
                actual_w, actual_h = im.size

            mismatches.append({
                "file_name": row["file_name"],
                "expected": f'{row["width"]}x{row["height"]}',
                "actual": f'{actual_w}x{actual_h}',
                "image_path": row["image_path"]
            })

print("=" * 80)
print("TACO DIMENSION MISMATCH ANALYSIS")
print("=" * 80)

print(f"Mismatch count: {len(mismatches)}")

print("\nTop expected -> actual dimension pairs:")
pairs = Counter((x["expected"], x["actual"]) for x in mismatches)

for (expected, actual), count in pairs.most_common(30):
    print(f"{count:4d} | expected {expected:>12} -> actual {actual}")

print("\nFirst 30 mismatches:")
print("-" * 80)

for x in mismatches[:30]:
    print(f'{x["file_name"]:30} expected={x["expected"]:>12} actual={x["actual"]:>12}')
