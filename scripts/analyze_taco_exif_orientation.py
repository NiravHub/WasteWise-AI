from pathlib import Path
from PIL import Image
import csv
from collections import Counter

manifest = Path(
    "results/dataset_audit/taco/taco_image_manifest_corrected.csv"
)

counter = Counter()

examples = []

with open(manifest, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["status"] != "DIMENSION_MISMATCH":
            continue

        path = Path(row["image_path"])

        with Image.open(path) as im:
            exif = im.getexif()
            orientation = exif.get(274, "NONE")

            actual = f"{im.width}x{im.height}"
            expected = f'{row["width"]}x{row["height"]}'

        counter[(orientation, expected, actual)] += 1

        if len(examples) < 30:
            examples.append(
                (row["file_name"], orientation, expected, actual)
            )

print("=" * 90)
print("TACO EXIF ORIENTATION ANALYSIS")
print("=" * 90)

print("\nOrientation distribution:")
for (orientation, expected, actual), count in counter.most_common():
    print(
        f"{count:4d} | EXIF={orientation!s:>4} | "
        f"expected={expected:>12} | actual={actual:>12}"
    )

print("\nFirst examples:")
print("-" * 90)

for filename, orientation, expected, actual in examples:
    print(
        f"{filename:32} "
        f"EXIF={orientation!s:>4} "
        f"expected={expected:>12} "
        f"actual={actual}"
    )
