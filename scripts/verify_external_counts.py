from pathlib import Path
from collections import Counter

ROOT = Path("data/external/garbage_object_detection")

CLASS_NAMES = [
    "BIODEGRADABLE",
    "CARDBOARD",
    "GLASS",
    "METAL",
    "PAPER",
    "PLASTIC",
]

total_lines = 0
valid_boxes = 0
invalid_boxes = 0
class_counts = Counter()
split_counts = Counter()
split_class_counts = {
    "train": Counter(),
    "valid": Counter(),
    "test": Counter(),
}

for split in ["train", "valid", "test"]:
    label_dir = ROOT / split / "labels"

    for label_path in sorted(label_dir.glob("*.txt")):
        lines = label_path.read_text(
            encoding="utf-8"
        ).splitlines()

        for line_no, line in enumerate(lines, start=1):
            line = line.strip()

            if not line:
                continue

            total_lines += 1

            parts = line.split()

            if len(parts) != 5:
                continue

            try:
                class_id = int(parts[0])
                xc, yc, w, h = map(float, parts[1:])
            except ValueError:
                continue

            if not (0 <= class_id < len(CLASS_NAMES)):
                continue

            # Reject zero-area boxes
            if w <= 0 or h <= 0:
                invalid_boxes += 1
                continue

            # Reject coordinates outside normalized image bounds
            if not (
                0 <= xc <= 1
                and 0 <= yc <= 1
                and 0 < w <= 1
                and 0 < h <= 1
            ):
                invalid_boxes += 1
                continue

            left = xc - w / 2
            right = xc + w / 2
            top = yc - h / 2
            bottom = yc + h / 2

            if (
                left < 0
                or right > 1
                or top < 0
                or bottom > 1
            ):
                invalid_boxes += 1
                continue

            valid_boxes += 1

            class_name = CLASS_NAMES[class_id]

            class_counts[class_name] += 1
            split_counts[split] += 1
            split_class_counts[split][class_name] += 1

print("=" * 70)
print("AUTHORITATIVE EXTERNAL DATASET ANNOTATION COUNT")
print("=" * 70)

print()
print(f"Total non-empty annotation lines : {total_lines}")
print(f"Valid boxes                      : {valid_boxes}")
print(f"Invalid boxes                    : {invalid_boxes}")
print(f"Valid + invalid                 : {valid_boxes + invalid_boxes}")

print()
print("CLASS DISTRIBUTION")
print("-" * 70)

for name in CLASS_NAMES:
    print(
        f"{name:20s} "
        f"{class_counts[name]:8d}"
    )

print()
print(
    f"CLASS TOTAL: {sum(class_counts.values())}"
)

print()
print("SPLIT DISTRIBUTION")
print("-" * 70)

for split in ["train", "valid", "test"]:
    print(
        f"{split:8s} "
        f"{split_counts[split]:8d}"
    )

print()
print(
    f"SPLIT TOTAL: {sum(split_counts.values())}"
)

print()
print("PER-SPLIT CLASS DISTRIBUTION")
print("-" * 70)

for split in ["train", "valid", "test"]:
    print()
    print(split.upper())

    for name in CLASS_NAMES:
        print(
            f"  {name:18s} "
            f"{split_class_counts[split][name]:8d}"
        )

print()
print("=" * 70)

if valid_boxes == sum(class_counts.values()):
    print("PASS — class counts reconcile with valid-box count.")
else:
    print("FAIL — class counts do not reconcile.")

if valid_boxes == sum(split_counts.values()):
    print("PASS — split counts reconcile with valid-box count.")
else:
    print("FAIL — split counts do not reconcile.")
