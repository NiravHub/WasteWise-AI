from pathlib import Path
from collections import Counter

ROOT = Path("data/processed/taco_detection")

print("=" * 75)
print("TACO FINAL STRUCTURAL AUDIT")
print("=" * 75)

splits = ["train", "val", "test"]

all_image_stems = {}
total_images = 0
total_labels = 0
total_boxes = 0

errors = []

for split in splits:
    image_dir = ROOT / "images" / split
    label_dir = ROOT / "labels" / split

    images = {p.stem for p in image_dir.iterdir() if p.is_file()}
    labels = {p.stem for p in label_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"}

    missing_labels = images - labels
    orphan_labels = labels - images

    if missing_labels:
        errors.append(f"{split}: {len(missing_labels)} images missing labels")

    if orphan_labels:
        errors.append(f"{split}: {len(orphan_labels)} orphan labels")

    all_image_stems[split] = images

    total_images += len(images)
    total_labels += len(labels)

    class_counts = Counter()
    split_boxes = 0
    empty_labels = 0

    for label_file in label_dir.glob("*.txt"):
        lines = [
            line.strip()
            for line in label_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        if not lines:
            empty_labels += 1

        for line_no, line in enumerate(lines, 1):
            parts = line.split()

            if len(parts) != 5:
                errors.append(
                    f"{split}/{label_file.name}: line {line_no} "
                    f"does not contain 5 fields"
                )
                continue

            try:
                cls = int(parts[0])
                xc, yc, w, h = map(float, parts[1:])
            except ValueError:
                errors.append(
                    f"{split}/{label_file.name}: line {line_no} "
                    f"contains non-numeric values"
                )
                continue

            if not 0 <= cls <= 22:
                errors.append(
                    f"{split}/{label_file.name}: invalid class ID {cls}"
                )

            if not (
                0 <= xc <= 1
                and 0 <= yc <= 1
                and 0 < w <= 1
                and 0 < h <= 1
            ):
                errors.append(
                    f"{split}/{label_file.name}: invalid YOLO bbox "
                    f"{xc} {yc} {w} {h}"
                )

            split_boxes += 1
            class_counts[cls] += 1

    total_boxes += split_boxes

    print()
    print(f"{split.upper()}")
    print("-" * 50)
    print(f"Images:        {len(images)}")
    print(f"Labels:        {len(labels)}")
    print(f"Boxes:         {split_boxes}")
    print(f"Empty labels:  {empty_labels}")
    print(f"Class IDs:     {sorted(class_counts)}")

    if missing_labels:
        print(f"Missing labels: {len(missing_labels)}")

    if orphan_labels:
        print(f"Orphan labels:  {len(orphan_labels)}")

print()
print("=" * 75)
print("CROSS-SPLIT CHECK")
print("=" * 75)

for i, a in enumerate(splits):
    for b in splits[i + 1:]:
        overlap = all_image_stems[a] & all_image_stems[b]

        print(
            f"{a} ∩ {b}: "
            f"{len(overlap)} overlapping image stems"
        )

        if overlap:
            errors.append(
                f"Cross-split overlap: {a}/{b} = {len(overlap)}"
            )

print()
print("=" * 75)
print("TOTALS")
print("=" * 75)

print(f"Images: {total_images}")
print(f"Labels: {total_labels}")
print(f"Boxes:  {total_boxes}")

print()
print("=" * 75)
print("FINAL STATUS")
print("=" * 75)

if errors:
    print(f"FAIL — {len(errors)} issue(s)")
    for error in errors[:50]:
        print(" -", error)
else:
    print("PASS — no structural or YOLO-format errors detected.")

print("=" * 75)
