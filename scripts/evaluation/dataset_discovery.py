from pathlib import Path
from collections import Counter

from dataset_registry import get_enabled_datasets


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_EXTENSIONS = {".txt"}


def find_images(path: Path):
    return [
        p for p in path.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def find_labels(path: Path):
    return [
        p for p in path.rglob("*")
        if p.is_file() and p.suffix.lower() in LABEL_EXTENSIONS
    ]


def discover_classification(config):
    root = config["path"]

    if not root.exists():
        return {
            "exists": False,
            "images": 0,
            "classes": {},
        }

    class_counts = {}

    for folder in root.iterdir():
        if not folder.is_dir():
            continue

        images = [
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if images:
            class_counts[folder.name] = len(images)

    return {
        "exists": True,
        "images": sum(class_counts.values()),
        "classes": class_counts,
    }


def discover_detection(config):
    root = config["path"]

    if not root.exists():
        return {
            "exists": False,
            "images": 0,
            "labels": 0,
            "paired": 0,
            "missing_labels": 0,
            "empty_labels": 0,
            "boxes": 0,
            "label_classes": {},
        }

    images = find_images(root)
    labels = find_labels(root)

    label_map = {
        label.stem: label
        for label in labels
    }

    paired = 0
    missing_labels = 0
    empty_labels = 0
    boxes = 0
    class_counts = Counter()

    for image in images:
        label = label_map.get(image.stem)

        if label is None:
            missing_labels += 1
            continue

        paired += 1

        try:
            lines = [
                line.strip()
                for line in label.read_text(
                    encoding="utf-8",
                    errors="ignore"
                ).splitlines()
                if line.strip()
            ]

            if not lines:
                empty_labels += 1
                continue

            boxes += len(lines)

            for line in lines:
                parts = line.split()

                if parts:
                    class_counts[parts[0]] += 1

        except Exception:
            empty_labels += 1

    return {
        "exists": True,
        "images": len(images),
        "labels": len(labels),
        "paired": paired,
        "missing_labels": missing_labels,
        "empty_labels": empty_labels,
        "boxes": boxes,
        "label_classes": dict(class_counts),
    }


def discover_negative(config):
    root = config["path"]

    if not root.exists():
        return {
            "exists": False,
            "images": 0,
        }

    images = find_images(root)

    return {
        "exists": True,
        "images": len(images),
    }


def discover_dataset(name, config):
    dataset_type = config["type"]

    if dataset_type == "classification":
        return discover_classification(config)

    if dataset_type == "detection":
        return discover_detection(config)

    if dataset_type == "negative":
        return discover_negative(config)

    raise ValueError(
        f"Unsupported dataset type: {dataset_type}"
    )


def print_class_counts(classes):
    if not classes:
        print("  Classes:      none")
        return

    for name, count in sorted(
        classes.items(),
        key=lambda item: (-item[1], item[0])
    ):
        print(f"    {name:<30} {count}")


def main():
    print("=" * 80)
    print("WASTEWISE DATASET DISCOVERY & VALIDATION")
    print("=" * 80)

    datasets = get_enabled_datasets()

    total_images = 0

    for name, config in datasets.items():

        print("\n" + "-" * 80)
        print(name)
        print("-" * 80)

        result = discover_dataset(name, config)

        if not result["exists"]:
            print("  STATUS:        MISSING PATH")
            continue

        if config["type"] == "classification":

            print("  Type:          Classification")
            print(f"  Images:        {result['images']}")
            print("  Classes:")

            print_class_counts(result["classes"])

            total_images += result["images"]

        elif config["type"] == "detection":

            print("  Type:          Detection")
            print(f"  Images:        {result['images']}")
            print(f"  Labels:        {result['labels']}")
            print(f"  Paired:        {result['paired']}")
            print(f"  Missing labels:{result['missing_labels']}")
            print(f"  Empty labels:  {result['empty_labels']}")
            print(f"  Bounding boxes:{result['boxes']}")

            print("  Label classes:")
            print_class_counts(result["label_classes"])

            total_images += result["images"]

        elif config["type"] == "negative":

            print("  Type:          Negative / OOD")
            print(f"  Images:        {result['images']}")

            total_images += result["images"]

    print("\n" + "=" * 80)
    print(f"TOTAL DISCOVERED IMAGES: {total_images}")
    print("=" * 80)


if __name__ == "__main__":
    main()