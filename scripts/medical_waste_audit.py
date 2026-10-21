from pathlib import Path
from collections import Counter, defaultdict
import json

ROOT = Path(__file__).resolve().parents[1]

DATASET = (
    ROOT
    / "data"
    / "external"
    / "medical_waste_4_0"
    / "Medical-Waste-4.0-Dataset"
)

OUTPUT = ROOT / "results" / "dataset_audit"
OUTPUT.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}


def count_images(folder):
    return sum(
        1
        for p in folder.rglob("*")
        if (
            p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
            and ".git" not in p.parts
        )
    )


def collect_leaf_folders(root):
    folders = []

    for p in root.rglob("*"):

        if not p.is_dir():
            continue

        if ".git" in p.parts:
            continue

        image_count = count_images(p)

        if image_count > 0:
            folders.append((p, image_count))

    return folders


def main():

    print("=" * 80)
    print("MEDICAL WASTE 4.0 — STRUCTURE & TAXONOMY AUDIT")
    print("=" * 80)

    print(f"\nDataset root:")
    print(DATASET)

    if not DATASET.exists():
        print("\nERROR: Dataset directory not found.")
        return

    # ------------------------------------------------------------
    # Top-level structure
    # ------------------------------------------------------------

    print("\n" + "-" * 80)
    print("TOP-LEVEL STRUCTURE")
    print("-" * 80)

    for item in sorted(DATASET.iterdir()):

        if item.name == ".git":
            continue

        if item.is_dir():
            print(f"[DIR ] {item.name}")
        else:
            print(f"[FILE] {item.name}")

    # ------------------------------------------------------------
    # Leaf folders containing images
    # ------------------------------------------------------------

    leaf_folders = collect_leaf_folders(DATASET)

    print("\n" + "-" * 80)
    print("IMAGE-CONTAINING FOLDERS")
    print("-" * 80)

    for folder, count in sorted(
        leaf_folders,
        key=lambda x: str(x[0]).lower()
    ):

        relative = folder.relative_to(DATASET)

        print(
            f"{str(relative):<70} "
            f"{count:>6,}"
        )

    # ------------------------------------------------------------
    # Determine dataset groups from path
    # ------------------------------------------------------------

    group_counts = defaultdict(Counter)

    for folder, count in leaf_folders:

        relative_parts = folder.relative_to(DATASET).parts

        if relative_parts:
            group = relative_parts[0]
        else:
            group = "ROOT"

        label = folder.name

        group_counts[group][label] += count

    print("\n" + "-" * 80)
    print("GROUP SUMMARY")
    print("-" * 80)

    for group, labels in sorted(group_counts.items()):

        total = sum(labels.values())

        print(f"\n{group}")
        print(f"Total images: {total:,}")

        for label, count in sorted(
            labels.items(),
            key=lambda x: (-x[1], x[0].lower())
        ):
            print(f"  {label:<45} {count:>6,}")

    # ------------------------------------------------------------
    # All image files by extension
    # ------------------------------------------------------------

    extension_counts = Counter()

    all_images = []

    for p in DATASET.rglob("*"):

        if (
            p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
            and ".git" not in p.parts
        ):

            extension_counts[p.suffix.lower()] += 1
            all_images.append(p)

    print("\n" + "-" * 80)
    print("IMAGE EXTENSIONS")
    print("-" * 80)

    for ext, count in extension_counts.most_common():
        print(f"{ext:<10} {count:>8,}")

    # ------------------------------------------------------------
    # Check for duplicate filenames
    # ------------------------------------------------------------

    filename_locations = defaultdict(list)

    for path in all_images:
        filename_locations[path.name].append(
            str(path.relative_to(DATASET))
        )

    repeated_names = {
        name: locations
        for name, locations in filename_locations.items()
        if len(locations) > 1
    }

    print("\n" + "-" * 80)
    print("FILENAME COLLISION CHECK")
    print("-" * 80)

    print(
        f"Unique filenames : {len(filename_locations):,}"
    )

    print(
        f"Repeated names   : {len(repeated_names):,}"
    )

    # ------------------------------------------------------------
    # Check likely label naming inconsistencies
    # ------------------------------------------------------------

    labels = sorted(
        {
            folder.name
            for folder, _ in leaf_folders
        },
        key=str.lower
    )

    print("\n" + "-" * 80)
    print("LABEL NAMES")
    print("-" * 80)

    for label in labels:
        print(f"  {label}")

    # ------------------------------------------------------------
    # Save report
    # ------------------------------------------------------------

    report = {
        "dataset": "Medical Waste 4.0",
        "dataset_root": str(DATASET),
        "total_images": len(all_images),
        "image_extensions": dict(extension_counts),
        "leaf_folders": [
            {
                "path": str(folder.relative_to(DATASET)),
                "image_count": count,
            }
            for folder, count in leaf_folders
        ],
        "group_summary": {
            group: dict(labels)
            for group, labels in group_counts.items()
        },
        "unique_filenames": len(filename_locations),
        "repeated_filename_count": len(repeated_names),
        "repeated_filenames": repeated_names,
        "labels": labels,
    }

    output = OUTPUT / "medical_waste_4_0_structure_audit.json"

    with open(output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 80)
    print("MEDICAL WASTE AUDIT COMPLETE")
    print("=" * 80)

    print(f"\nSaved:")
    print(output)


if __name__ == "__main__":
    main()