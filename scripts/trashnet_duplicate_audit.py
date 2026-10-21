from pathlib import Path
from collections import defaultdict, Counter
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "external" / "trashnet_plus_plus"
OUTPUT = ROOT / "results" / "dataset_audit"
OUTPUT.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def get_images(root):
    return [
        p for p in root.rglob("*")
        if (
            p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
            and ".git" not in p.parts
        )
    ]


def get_source(path):
    parts = path.parts

    if "Classwise Formet" in parts:
        return "classwise"

    if "YOLO Formet" in parts:
        return "yolo"

    return "other"


def get_classwise_label(path):
    parts = path.parts

    if "Classwise Formet" in parts:
        index = parts.index("Classwise Formet")

        if index + 1 < len(parts):
            return parts[index + 1]

    return None


def get_yolo_split(path):
    parts = path.parts

    if "YOLO Formet" in parts:
        index = parts.index("YOLO Formet")

        if index + 1 < len(parts):
            return parts[index + 1]

    return None


def main():

    print("=" * 80)
    print("TRASHNET++ DUPLICATE PROVENANCE AUDIT")
    print("=" * 80)

    images = get_images(DATASET)

    print(f"\nTotal image files: {len(images):,}")

    hashes = defaultdict(list)

    for i, path in enumerate(images, 1):

        if i % 500 == 0 or i == len(images):
            print(
                f"\rHashed {i:,}/{len(images):,}",
                end="",
                flush=True
            )

        digest = sha256_file(path)
        hashes[digest].append(path)

    print("\n")

    duplicate_groups = {
        digest: paths
        for digest, paths in hashes.items()
        if len(paths) > 1
    }

    print(f"Unique images       : {len(hashes):,}")
    print(f"Duplicate groups    : {len(duplicate_groups):,}")

    duplicate_files = sum(len(x) for x in duplicate_groups.values())

    print(f"Files in duplicates : {duplicate_files:,}")

    # ------------------------------------------------------------------
    # Analyze duplicate groups
    # ------------------------------------------------------------------

    duplicate_type_counts = Counter()
    cross_class_groups = []
    classwise_yolo_groups = []
    internal_classwise_groups = []
    internal_yolo_groups = []
    other_groups = []

    for digest, paths in duplicate_groups.items():

        sources = set(get_source(p) for p in paths)

        class_labels = set()
        yolo_splits = set()

        for path in paths:

            label = get_classwise_label(path)

            if label:
                class_labels.add(label)

            split = get_yolo_split(path)

            if split:
                yolo_splits.add(split)

        # Classwise <-> YOLO
        if "classwise" in sources and "yolo" in sources:
            duplicate_type_counts["classwise_vs_yolo"] += 1
            classwise_yolo_groups.append((digest, paths))

        # Multiple classwise labels
        elif len(class_labels) > 1:
            duplicate_type_counts["cross_class_classwise"] += 1
            cross_class_groups.append(
                {
                    "sha256": digest,
                    "labels": sorted(class_labels),
                    "files": [str(p) for p in paths],
                }
            )

        # Multiple copies within same class
        elif "classwise" in sources:
            duplicate_type_counts["within_classwise"] += 1
            internal_classwise_groups.append((digest, paths))

        # Multiple YOLO copies
        elif "yolo" in sources:
            duplicate_type_counts["within_yolo"] += 1
            internal_yolo_groups.append((digest, paths))

        else:
            duplicate_type_counts["other"] += 1
            other_groups.append((digest, paths))

    print("\nDuplicate group types:")

    for key, value in duplicate_type_counts.most_common():
        print(f"  {key:<30} {value:>6,}")

    # ------------------------------------------------------------------
    # Classwise -> YOLO mapping
    # ------------------------------------------------------------------

    classwise_to_yolo = []

    for digest, paths in classwise_yolo_groups:

        classwise_labels = sorted(
            {
                label
                for path in paths
                if (label := get_classwise_label(path))
            }
        )

        yolo_splits = sorted(
            {
                split
                for path in paths
                if (split := get_yolo_split(path))
            }
        )

        classwise_to_yolo.append(
            {
                "sha256": digest,
                "classwise_labels": classwise_labels,
                "yolo_splits": yolo_splits,
                "files": [str(p) for p in paths],
            }
        )

    # ------------------------------------------------------------------
    # Unique images by classwise label
    # ------------------------------------------------------------------

    unique_classwise_hashes = defaultdict(set)

    for digest, paths in hashes.items():

        for path in paths:

            label = get_classwise_label(path)

            if label:
                unique_classwise_hashes[label].add(digest)

    # ------------------------------------------------------------------
    # Print classwise unique counts
    # ------------------------------------------------------------------

    print("\nUnique image contents by Classwise label:")

    for label, digests in sorted(
        unique_classwise_hashes.items(),
        key=lambda x: (-len(x[1]), x[0])
    ):
        print(f"  {label:<25} {len(digests):>6,}")

    # ------------------------------------------------------------------
    # Save report
    # ------------------------------------------------------------------

    report = {
        "dataset": "TrashNet++",
        "total_image_files": len(images),
        "unique_image_contents": len(hashes),
        "duplicate_groups": len(duplicate_groups),
        "files_in_duplicate_groups": duplicate_files,
        "duplicate_group_types": dict(duplicate_type_counts),

        "unique_images_by_classwise_label": {
            label: len(digests)
            for label, digests in unique_classwise_hashes.items()
        },

        "cross_class_duplicate_groups": cross_class_groups,

        "classwise_yolo_duplicate_groups": classwise_to_yolo,

        "within_classwise_duplicate_groups": [
            {
                "sha256": digest,
                "files": [str(p) for p in paths],
            }
            for digest, paths in internal_classwise_groups
        ],

        "within_yolo_duplicate_groups": [
            {
                "sha256": digest,
                "files": [str(p) for p in paths],
            }
            for digest, paths in internal_yolo_groups
        ],

        "other_duplicate_groups": [
            {
                "sha256": digest,
                "files": [str(p) for p in paths],
            }
            for digest, paths in other_groups
        ],
    }

    output = OUTPUT / "trashnet_duplicate_provenance.json"

    with open(output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)

    print(f"\nReport saved to:")
    print(output)


if __name__ == "__main__":
    main()