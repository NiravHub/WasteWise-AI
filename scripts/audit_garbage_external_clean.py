from pathlib import Path
from PIL import Image
from collections import Counter
import hashlib

ROOT = Path(__file__).resolve().parents[1]

DATASET = ROOT / "data" / "processed" / "garbage_external_clean"

SPLITS = ["train", "valid", "test"]

CLASS_NAMES = [
    "BIODEGRADABLE",
    "CARDBOARD",
    "GLASS",
    "METAL",
    "PAPER",
    "PLASTIC",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


total_images = 0
total_boxes = 0
invalid_images = 0
invalid_boxes = 0
missing_labels = 0
orphan_labels = 0

class_counts = Counter()
split_counts = Counter()

hashes = {}

print("=" * 70)
print("CLEAN EXTERNAL DATASET AUDIT")
print("=" * 70)

for split in SPLITS:

    image_dir = DATASET / split / "images"
    label_dir = DATASET / split / "labels"

    images = [
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]

    labels = list(label_dir.glob("*.txt"))

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    missing = image_stems - label_stems
    orphan = label_stems - image_stems

    missing_labels += len(missing)
    orphan_labels += len(orphan)

    print(f"\n{split.upper()}")

    print(f"Images: {len(images)}")
    print(f"Labels: {len(labels)}")
    print(f"Missing labels: {len(missing)}")
    print(f"Orphan labels: {len(orphan)}")

    split_box_count = 0

    for image_path in images:

        total_images += 1

        try:
            with Image.open(image_path) as im:
                im.verify()

            with Image.open(image_path) as im:
                width, height = im.size

            digest = sha256(image_path)

            hashes.setdefault(digest, []).append(image_path)

        except Exception as exc:

            invalid_images += 1

            print(
                f"INVALID IMAGE: {image_path.name} -> {exc}"
            )

            continue

        label_path = label_dir / f"{image_path.stem}.txt"

        with open(label_path, "r", encoding="utf-8") as f:

            for line_number, line in enumerate(f, 1):

                line = line.strip()

                if not line:
                    continue

                parts = line.split()

                if len(parts) != 5:

                    invalid_boxes += 1

                    print(
                        f"INVALID LABEL FORMAT: "
                        f"{split}/{label_path.name}:{line_number}"
                    )

                    continue

                try:
                    class_id = int(float(parts[0]))
                    x = float(parts[1])
                    y = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])

                except ValueError:

                    invalid_boxes += 1

                    print(
                        f"INVALID NUMERIC LABEL: "
                        f"{split}/{label_path.name}:{line_number}"
                    )

                    continue

                if not 0 <= class_id < len(CLASS_NAMES):

                    invalid_boxes += 1

                    print(
                        f"INVALID CLASS ID: "
                        f"{split}/{label_path.name}:{line_number}"
                    )

                    continue

                if w <= 0 or h <= 0:

                    invalid_boxes += 1

                    print(
                        f"INVALID AREA: "
                        f"{split}/{label_path.name}:{line_number}"
                    )

                    continue

                if not (
                    0 <= x <= 1
                    and 0 <= y <= 1
                    and 0 < w <= 1
                    and 0 < h <= 1
                ):

                    invalid_boxes += 1

                    print(
                        f"INVALID YOLO COORDINATES: "
                        f"{split}/{label_path.name}:{line_number}"
                    )

                    continue

                left = x - w / 2
                right = x + w / 2
                top = y - h / 2
                bottom = y + h / 2

                if (
                    left < 0
                    or right > 1
                    or top < 0
                    or bottom > 1
                ):

                    invalid_boxes += 1

                    print(
                        f"OUT-OF-BOUNDS BOX: "
                        f"{split}/{label_path.name}:{line_number}"
                    )

                    continue

                class_counts[CLASS_NAMES[class_id]] += 1
                split_counts[split] += 1
                split_box_count += 1
                total_boxes += 1

    print(f"Valid boxes: {split_box_count}")


duplicate_groups = [
    paths
    for paths in hashes.values()
    if len(paths) > 1
]

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

for name in CLASS_NAMES:
    print(
        f"{name:<20} {class_counts[name]:>8}"
    )

print("\n" + "=" * 70)
print("FINAL AUDIT")
print("=" * 70)

print(f"Total images:          {total_images}")
print(f"Total valid boxes:     {total_boxes}")
print(f"Invalid images:        {invalid_images}")
print(f"Invalid boxes:         {invalid_boxes}")
print(f"Missing labels:        {missing_labels}")
print(f"Orphan labels:         {orphan_labels}")
print(f"Duplicate groups:      {len(duplicate_groups)}")

print("\nSplit box counts:")

for split in SPLITS:
    print(
        f"{split:<10} {split_counts[split]}"
    )

if (
    invalid_images == 0
    and invalid_boxes == 0
    and missing_labels == 0
    and orphan_labels == 0
):
    print("\nPASS — clean external dataset is structurally valid.")
else:
    print("\nFAIL — structural issues remain.")