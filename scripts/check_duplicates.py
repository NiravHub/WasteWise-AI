from pathlib import Path
from collections import defaultdict
import hashlib

# ============================================================
# WasteWise AI - Duplicate Image Analysis
# ============================================================

DATASET_DIR = Path("data/raw/garbage-dataset")

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def calculate_sha256(file_path: Path) -> str:
    """Calculate SHA-256 hash of a file."""
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            sha256.update(chunk)

    return sha256.hexdigest()


print("=" * 70)
print("WASTEWISE AI - DUPLICATE IMAGE ANALYSIS")
print("=" * 70)

if not DATASET_DIR.exists():
    print(f"\nERROR: Dataset not found:")
    print(DATASET_DIR.resolve())
    raise SystemExit(1)

# ------------------------------------------------------------
# Scan files
# ------------------------------------------------------------

image_files = []

for file_path in DATASET_DIR.rglob("*"):
    if (
        file_path.is_file()
        and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
    ):
        image_files.append(file_path)

print(f"\nImages scanned: {len(image_files)}")

# ------------------------------------------------------------
# Calculate hashes
# ------------------------------------------------------------

hash_groups = defaultdict(list)

print("\nCalculating SHA-256 hashes...")

for index, file_path in enumerate(image_files, start=1):

    file_hash = calculate_sha256(file_path)
    hash_groups[file_hash].append(file_path)

    if index % 1000 == 0:
        print(f"  Processed {index}/{len(image_files)}")

# ------------------------------------------------------------
# Find exact duplicates
# ------------------------------------------------------------

duplicate_groups = [
    paths
    for paths in hash_groups.values()
    if len(paths) > 1
]

duplicate_images = sum(
    len(paths) - 1
    for paths in duplicate_groups
)

print("\n" + "=" * 70)
print("EXACT DUPLICATE RESULTS")
print("=" * 70)

print(f"\nUnique image hashes : {len(hash_groups)}")
print(f"Duplicate groups    : {len(duplicate_groups)}")
print(f"Duplicate images    : {duplicate_images}")

# ------------------------------------------------------------
# Display duplicate groups
# ------------------------------------------------------------

if duplicate_groups:

    print("\nDuplicate groups found:\n")

    for group_number, paths in enumerate(duplicate_groups, start=1):

        print(f"Group {group_number}:")

        for path in paths:
            print(f"  {path}")

        print()

else:
    print("\nNo exact duplicate images found.")

# ------------------------------------------------------------
# Check duplicates across different classes
# ------------------------------------------------------------

cross_class_duplicates = []

for paths in duplicate_groups:

    classes = {
        path.parent.name
        for path in paths
    }

    if len(classes) > 1:
        cross_class_duplicates.append(
            {
                "classes": sorted(classes),
                "paths": paths,
            }
        )

print("=" * 70)
print("CROSS-CLASS DUPLICATES")
print("=" * 70)

print(
    f"\nDuplicate groups appearing in multiple classes: "
    f"{len(cross_class_duplicates)}"
)

if cross_class_duplicates:

    for group_number, group in enumerate(
        cross_class_duplicates,
        start=1
    ):

        print(f"\nGroup {group_number}")
        print(f"Classes: {', '.join(group['classes'])}")

        for path in group["paths"]:
            print(f"  {path}")

else:
    print("\nNo cross-class exact duplicates found.")

# ------------------------------------------------------------
# Final status
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DUPLICATE ANALYSIS COMPLETE")
print("=" * 70)

print("\nNo files were modified or deleted.")