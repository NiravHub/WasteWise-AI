from pathlib import Path
from collections import Counter
from PIL import Image

# ============================================================
# WasteWise AI - Dataset Audit
# ============================================================

DATASET_DIR = Path("data/raw/garbage-dataset")

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

print("=" * 70)
print("WASTEWISE AI - DATASET AUDIT")
print("=" * 70)

if not DATASET_DIR.exists():
    print(f"\nERROR: Dataset not found at:")
    print(DATASET_DIR.resolve())
    raise SystemExit(1)

print(f"\nDataset path:")
print(DATASET_DIR.resolve())

# ------------------------------------------------------------
# Discover class directories
# ------------------------------------------------------------

class_dirs = sorted(
    [p for p in DATASET_DIR.iterdir() if p.is_dir()]
)

print(f"\nNumber of class directories: {len(class_dirs)}")

if not class_dirs:
    print("\nERROR: No class directories found.")
    raise SystemExit(1)

print("\nClasses:")
for directory in class_dirs:
    print(f"  - {directory.name}")

# ------------------------------------------------------------
# Inspect images
# ------------------------------------------------------------

class_counts = Counter()
extension_counts = Counter()
dimension_counts = Counter()

invalid_images = []
unsupported_files = []

total_files = 0
total_images = 0

print("\n" + "=" * 70)
print("SCANNING DATASET")
print("=" * 70)

for class_dir in class_dirs:

    print(f"\nScanning: {class_dir.name}")

    for file_path in class_dir.rglob("*"):

        if not file_path.is_file():
            continue

        total_files += 1

        extension = file_path.suffix.lower()

        if extension not in SUPPORTED_EXTENSIONS:
            unsupported_files.append(file_path)
            continue

        extension_counts[extension] += 1

        try:
            with Image.open(file_path) as img:

                # Force actual image verification
                img.verify()

            # Re-open after verify() because verify() invalidates
            # the image object.
            with Image.open(file_path) as img:

                width, height = img.size
                dimension_counts[(width, height)] += 1

            class_counts[class_dir.name] += 1
            total_images += 1

        except Exception as error:

            invalid_images.append(
                {
                    "path": str(file_path),
                    "error": str(error),
                }
            )

# ------------------------------------------------------------
# Dataset summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATASET SUMMARY")
print("=" * 70)

print(f"\nTotal files discovered : {total_files}")
print(f"Valid images           : {total_images}")
print(f"Invalid/corrupt images : {len(invalid_images)}")
print(f"Unsupported files      : {len(unsupported_files)}")

# ------------------------------------------------------------
# Class distribution
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

for class_name in sorted(class_counts):
    print(f"{class_name:<20} {class_counts[class_name]:>6}")

# ------------------------------------------------------------
# File extensions
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("IMAGE FORMATS")
print("=" * 70)

for extension, count in sorted(extension_counts.items()):
    print(f"{extension:<10} {count:>6}")

# ------------------------------------------------------------
# Dimensions
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("IMAGE DIMENSIONS")
print("=" * 70)

for (width, height), count in dimension_counts.most_common(20):
    print(f"{width} x {height:<6} : {count}")

if len(dimension_counts) > 20:
    print(
        f"\n... and {len(dimension_counts) - 20} additional dimension combinations."
    )

# ------------------------------------------------------------
# Class imbalance
# ------------------------------------------------------------

if class_counts:

    largest_class = max(class_counts.values())
    smallest_class = min(class_counts.values())

    imbalance_ratio = (
        largest_class / smallest_class
        if smallest_class > 0
        else float("inf")
    )

    print("\n" + "=" * 70)
    print("CLASS BALANCE")
    print("=" * 70)

    print(f"\nLargest class : {largest_class}")
    print(f"Smallest class: {smallest_class}")
    print(f"Imbalance ratio: {imbalance_ratio:.2f}x")

# ------------------------------------------------------------
# Invalid images
# ------------------------------------------------------------

if invalid_images:

    print("\n" + "=" * 70)
    print("INVALID / CORRUPT IMAGES")
    print("=" * 70)

    for item in invalid_images[:20]:
        print(f"\n{item['path']}")
        print(f"Error: {item['error']}")

    if len(invalid_images) > 20:
        print(
            f"\n... and {len(invalid_images) - 20} more invalid images."
        )

# ------------------------------------------------------------
# Unsupported files
# ------------------------------------------------------------

if unsupported_files:

    print("\n" + "=" * 70)
    print("UNSUPPORTED FILES")
    print("=" * 70)

    for file_path in unsupported_files[:20]:
        print(file_path)

    if len(unsupported_files) > 20:
        print(
            f"\n... and {len(unsupported_files) - 20} more unsupported files."
        )

# ------------------------------------------------------------
# Final status
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)

print("\nNo files were modified or deleted.")
print("Dataset remains untouched.")