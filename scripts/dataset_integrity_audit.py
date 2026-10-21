from pathlib import Path
from collections import Counter
import hashlib
import json

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "garbage_dataset": ROOT / "data" / "raw" / "garbage-dataset",
    "coco": ROOT / "data" / "external" / "coco",
    "taco": ROOT / "data" / "external" / "taco",
    "trashnet_plus_plus": ROOT / "data" / "external" / "trashnet_plus_plus",
    "medical_waste_4_0": ROOT / "data" / "external" / "medical_waste_4_0",
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}

OUTPUT = ROOT / "results" / "dataset_audit"
OUTPUT.mkdir(parents=True, exist_ok=True)


def get_images(root):
    if not root.exists():
        return []

    return [
        p
        for p in root.rglob("*")
        if (
            p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
            and ".git" not in p.parts
        )
    ]


def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def audit_dataset(name, root):
    print("\n" + "=" * 80)
    print(f"AUDITING: {name}")
    print("=" * 80)

    images = get_images(root)

    print(f"Images found : {len(images):,}")

    if not images:
        return {
            "dataset": name,
            "path": str(root),
            "images_found": 0,
        }

    corrupt = []
    valid = []

    dimensions = Counter()
    formats = Counter()
    modes = Counter()
    aspect_ratios = Counter()

    zero_byte = []
    hashes = {}

    for index, path in enumerate(images, start=1):

        if index % 500 == 0 or index == len(images):
            print(
                f"\rProcessed {index:,}/{len(images):,}",
                end="",
                flush=True,
            )

        # ----------------------------------------------------
        # Zero-byte check
        # ----------------------------------------------------

        try:
            size = path.stat().st_size

            if size == 0:
                zero_byte.append(str(path))
                continue

        except Exception:
            zero_byte.append(str(path))
            continue

        # ----------------------------------------------------
        # Image integrity
        # ----------------------------------------------------

        try:
            with Image.open(path) as img:

                # Force PIL to verify the image structure.
                img.verify()

            # Re-open after verify because verify() invalidates
            # the image object.
            with Image.open(path) as img:

                width, height = img.size
                mode = img.mode
                fmt = img.format

            valid.append(str(path))

            dimensions[f"{width}x{height}"] += 1
            formats[fmt or path.suffix.lower()] += 1
            modes[mode] += 1

            if height != 0:
                ratio = round(width / height, 3)
                aspect_ratios[str(ratio)] += 1

            # ------------------------------------------------
            # Exact SHA-256 duplicate detection
            # ------------------------------------------------

            digest = sha256_file(path)

            hashes.setdefault(digest, []).append(str(path))

        except Exception as e:
            corrupt.append({
                "path": str(path),
                "error": str(e),
            })

    print()

    # --------------------------------------------------------
    # Duplicate groups
    # --------------------------------------------------------

    duplicate_groups = {
        digest: paths
        for digest, paths in hashes.items()
        if len(paths) > 1
    }

    duplicate_images = sum(
        len(paths)
        for paths in duplicate_groups.values()
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report = {
        "dataset": name,
        "path": str(root),
        "images_found": len(images),
        "valid_images": len(valid),
        "corrupt_images": len(corrupt),
        "zero_byte_files": len(zero_byte),
        "unique_sha256_hashes": len(hashes),
        "duplicate_groups": len(duplicate_groups),
        "duplicate_images": duplicate_images,
        "formats": dict(formats),
        "modes": dict(modes),
        "dimensions": dict(dimensions),
        "aspect_ratios": dict(aspect_ratios),
        "corrupt_files": corrupt,
        "zero_byte_files_list": zero_byte,
        "duplicate_groups_detail": duplicate_groups,
    }

    output_file = OUTPUT / f"{name}_integrity.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nValid images       : {len(valid):,}")
    print(f"Corrupt images     : {len(corrupt):,}")
    print(f"Zero-byte files    : {len(zero_byte):,}")
    print(f"Unique SHA-256     : {len(hashes):,}")
    print(f"Duplicate groups   : {len(duplicate_groups):,}")
    print(f"Duplicate images   : {duplicate_images:,}")

    print("\nFormats:")

    for key, value in formats.most_common():
        print(f"  {key:<10} {value:>8,}")

    print("\nColor modes:")

    for key, value in modes.most_common():
        print(f"  {key:<10} {value:>8,}")

    print("\nMost common dimensions:")

    for key, value in dimensions.most_common(15):
        print(f"  {key:<15} {value:>8,}")

    print(f"\nSaved: {output_file}")

    return report


def main():

    print("=" * 80)
    print("WASTEWISE AI — DATASET INTEGRITY AUDIT")
    print("=" * 80)

    all_reports = {}

    for name, root in DATASETS.items():
        all_reports[name] = audit_dataset(name, root)

    combined_file = OUTPUT / "integrity_summary.json"

    with open(combined_file, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=2)

    print("\n" + "=" * 80)
    print("INTEGRITY AUDIT COMPLETE")
    print("=" * 80)

    print(f"Reports saved to:")
    print(OUTPUT)


if __name__ == "__main__":
    main()