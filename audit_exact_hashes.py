from pathlib import Path
import hashlib
from collections import defaultdict

ROOT = Path("data")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def collect(folder):
    files = [
        p for p in Path(folder).rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]

    hashes = defaultdict(list)

    for i, path in enumerate(files, 1):
        hashes[sha256(path)].append(path)

        if i % 1000 == 0:
            print(f"  hashed {i}/{len(files)}")

    return files, hashes

def report(name, folder):
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)
    print("Folder:", folder)

    files, hashes = collect(folder)

    duplicate_groups = {
        h: paths for h, paths in hashes.items()
        if len(paths) > 1
    }

    duplicate_files = sum(len(v) for v in duplicate_groups.values())

    print("Images:", len(files))
    print("Unique SHA256:", len(hashes))
    print("Duplicate groups:", len(duplicate_groups))
    print("Duplicate files:", duplicate_files)

    if duplicate_groups:
        print("\nDUPLICATES FOUND:")
        for h, paths in list(duplicate_groups.items())[:10]:
            print("\nSHA256:", h)
            for p in paths:
                print("  ", p)

    return hashes


# ------------------------------------------------------------
# 1. Original classifier dataset
# ------------------------------------------------------------

original = ROOT / "raw" / "garbage-dataset"

if original.exists():
    report(
        "ORIGINAL CLASSIFIER DATASET",
        original,
    )


# ------------------------------------------------------------
# 2. Expanded classifier data
# ------------------------------------------------------------

expanded = ROOT / "processed"

if expanded.exists():
    print("\n" + "=" * 70)
    print("PROCESSED DATASET INVENTORY")
    print("=" * 70)

    for folder in sorted(expanded.iterdir()):
        if folder.is_dir():
            count = sum(
                1
                for p in folder.rglob("*")
                if p.is_file()
                and p.suffix.lower() in IMAGE_EXTS
            )

            if count:
                print(f"{folder}: {count}")


# ------------------------------------------------------------
# 3. Detector datasets
# ------------------------------------------------------------

detector_candidates = [
    ROOT / "processed" / "taco_practical",
    ROOT / "processed" / "garbage_external_wastewise",
]

for folder in detector_candidates:
    if not folder.exists():
        continue

    print("\n" + "=" * 70)
    print("DETECTOR DATASET")
    print("=" * 70)
    print(folder)

    for split in ["train", "val", "test"]:
        image_dir = folder / split / "images"

        if not image_dir.exists():
            continue

        files, hashes = collect(image_dir)

        print(
            f"{split}: "
            f"{len(files)} images, "
            f"{len(hashes)} unique SHA256"
        )


print("\n" + "=" * 70)
print("EXACT HASH INVENTORY COMPLETE")
print("=" * 70)
