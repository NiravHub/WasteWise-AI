from pathlib import Path
import yaml
from PIL import Image

ROOT = Path("data/processed/taco_detection_robust")
DATA_YAML = ROOT / "data.yaml"

print("=" * 80)
print("TACO ROBUST 33-CLASS DATASET AUDIT")
print("=" * 80)

with open(DATA_YAML, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

names = data["names"]

if isinstance(names, dict):
    names = [names[k] for k in sorted(names, key=lambda x: int(x))]

errors = []

if len(names) != 33:
    errors.append(f"Expected 33 classes, found {len(names)}")

total_images = 0
total_boxes = 0
all_stems = {}
class_counts = [0] * len(names)

for split in ["train", "val", "test"]:

    image_dir = ROOT / "images" / split
    label_dir = ROOT / "labels" / split

    images = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    labels = sorted(label_dir.glob("*.txt"))

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    missing_labels = image_stems - label_stems
    orphan_labels = label_stems - image_stems

    if missing_labels:
        errors.append(
            f"{split}: {len(missing_labels)} images missing labels"
        )

    if orphan_labels:
        errors.append(
            f"{split}: {len(orphan_labels)} orphan labels"
        )

    split_boxes = 0

    for img in images:
        try:
            with Image.open(img) as im:
                im.verify()
        except Exception as e:
            errors.append(f"Corrupt image: {img} -> {e}")

    for label in labels:

        with open(label, "r", encoding="utf-8") as f:
            lines = [x.strip() for x in f if x.strip()]

        for line_no, line in enumerate(lines, 1):

            parts = line.split()

            if len(parts) != 5:
                errors.append(
                    f"{label}: line {line_no}: expected 5 values, got {len(parts)}"
                )
                continue

            try:
                cls = int(parts[0])
                vals = [float(x) for x in parts[1:]]
            except ValueError:
                errors.append(
                    f"{label}: line {line_no}: non-numeric value"
                )
                continue

            if cls < 0 or cls >= len(names):
                errors.append(
                    f"{label}: line {line_no}: invalid class {cls}"
                )
                continue

            x, y, w, h = vals

            if not all(0.0 <= v <= 1.0 for v in vals):
                errors.append(
                    f"{label}: line {line_no}: value outside [0,1]"
                )

            if w <= 0 or h <= 0:
                errors.append(
                    f"{label}: line {line_no}: non-positive box"
                )

            class_counts[cls] += 1
            split_boxes += 1

    total_images += len(images)
    total_boxes += split_boxes

    for stem in image_stems:

        if stem in all_stems:
            errors.append(
                f"Cross-split overlap: {stem} "
                f"({all_stems[stem]} and {split})"
            )
        else:
            all_stems[stem] = split

    print(f"{split:5s}: {len(images):4d} images, {split_boxes:4d} boxes")

print()
print("SUMMARY")
print("-" * 80)
print(f"Classes:          {len(names)}")
print(f"Images:           {total_images}")
print(f"Boxes:            {total_boxes}")
print("Expected images:  1500")
print("Expected boxes:   4784")
print()

print("CLASS DISTRIBUTION")
print("-" * 80)

for i, (name, count) in enumerate(zip(names, class_counts)):
    print(f"{i:2d}  {name:25s} {count:4d}")

print()
print("AUDIT RESULT")
print("-" * 80)

if total_images != 1500:
    errors.append(f"Expected 1500 images, found {total_images}")

if total_boxes != 4784:
    errors.append(f"Expected 4784 boxes, found {total_boxes}")

if errors:

    print(f"FAIL - {len(errors)} issue(s)")

    for error in errors[:50]:
        print(" -", error)

    if len(errors) > 50:
        print(f" ... and {len(errors) - 50} more")

else:

    print("PASS - robust 33-class dataset is structurally valid.")
    print("PASS - 1500 images verified.")
    print("PASS - 4784 boxes verified.")
    print("PASS - YOLO coordinates valid.")
    print("PASS - no cross-split leakage detected.")
