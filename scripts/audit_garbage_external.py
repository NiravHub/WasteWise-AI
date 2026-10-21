from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image
import hashlib
import ast
import csv

ROOT = Path("data/external/garbage_object_detection")
OUT = Path("results/dataset_audit/external_garbage")
OUT.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "valid", "test"]

# ------------------------------------------------------------
# Read data.yaml
# ------------------------------------------------------------

yaml_path = ROOT / "data.yaml"
yaml_text = yaml_path.read_text(encoding="utf-8")

names_line = next(
    line for line in yaml_text.splitlines()
    if line.strip().startswith("names:")
)

names = ast.literal_eval(names_line.split(":", 1)[1].strip())

print("=" * 70)
print("EXTERNAL GARBAGE DATASET AUDIT")
print("=" * 70)
print(f"Root: {ROOT}")
print(f"Classes: {len(names)}")
print(f"Names: {names}")
print()

# ------------------------------------------------------------
# Counters
# ------------------------------------------------------------

total_images = 0
total_labels = 0

valid_images = 0
corrupt_images = []

missing_labels = []
orphan_labels = []

invalid_boxes = []
invalid_class_ids = []

class_counts = Counter()
images_per_class = Counter()

image_records = []
hash_to_files = defaultdict(list)

split_summary = []

# ------------------------------------------------------------
# Process each split
# ------------------------------------------------------------

for split in SPLITS:

    image_dir = ROOT / split / "images"
    label_dir = ROOT / split / "labels"

    if not image_dir.exists():
        print(f"WARNING: missing image directory: {image_dir}")
        continue

    if not label_dir.exists():
        print(f"WARNING: missing label directory: {label_dir}")
        continue

    image_files = sorted(
        [
            p for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in
            {".jpg", ".jpeg", ".png", ".webp"}
        ]
    )

    label_files = sorted(
        [
            p for p in label_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".txt"
        ]
    )

    image_stems = {p.stem: p for p in image_files}
    label_stems = {p.stem: p for p in label_files}

    split_missing_labels = sorted(
        set(image_stems) - set(label_stems)
    )

    split_orphan_labels = sorted(
        set(label_stems) - set(image_stems)
    )

    missing_labels.extend(
        [(split, stem) for stem in split_missing_labels]
    )

    orphan_labels.extend(
        [(split, stem) for stem in split_orphan_labels]
    )

    split_boxes = 0
    split_invalid = 0

    for img_path in image_files:

        total_images += 1

        # Read image
        try:
            with Image.open(img_path) as im:
                width, height = im.size
                im.verify()

            valid_images += 1

        except Exception as e:
            corrupt_images.append(
                (split, str(img_path), str(e))
            )
            continue

        # Hash
        try:
            h = hashlib.sha256()
            with open(img_path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)

            sha = h.hexdigest()
            hash_to_files[sha].append(str(img_path))

        except Exception:
            sha = ""

        label_path = label_stems.get(img_path.stem)

        if label_path is None:
            continue

        try:
            lines = label_path.read_text(
                encoding="utf-8"
            ).splitlines()

        except Exception as e:
            invalid_boxes.append(
                (split, img_path.name, "label_read_error", str(e))
            )
            continue

        image_class_set = set()

        for line_no, line in enumerate(lines, start=1):

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:
                invalid_boxes.append(
                    (
                        split,
                        img_path.name,
                        "wrong_field_count",
                        f"line={line_no}, fields={len(parts)}"
                    )
                )
                split_invalid += 1
                continue

            try:
                class_id = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:])

            except Exception as e:
                invalid_boxes.append(
                    (
                        split,
                        img_path.name,
                        "parse_error",
                        f"line={line_no}, {e}"
                    )
                )
                split_invalid += 1
                continue

            if class_id < 0 or class_id >= len(names):
                invalid_class_ids.append(
                    (
                        split,
                        img_path.name,
                        class_id,
                        line_no
                    )
                )
                split_invalid += 1
                continue

            if (
                not (0.0 <= xc <= 1.0)
                or not (0.0 <= yc <= 1.0)
                or not (0.0 < bw <= 1.0)
                or not (0.0 < bh <= 1.0)
            ):
                invalid_boxes.append(
                    (
                        split,
                        img_path.name,
                        "invalid_normalized_bbox",
                        f"line={line_no}: {xc} {yc} {bw} {bh}"
                    )
                )
                split_invalid += 1
                continue

            # Check box boundaries
            left = xc - bw / 2
            right = xc + bw / 2
            top = yc - bh / 2
            bottom = yc + bh / 2

            if left < 0 or right > 1 or top < 0 or bottom > 1:
                invalid_boxes.append(
                    (
                        split,
                        img_path.name,
                        "bbox_out_of_bounds",
                        f"line={line_no}: {xc} {yc} {bw} {bh}"
                    )
                )
                split_invalid += 1
                continue

            class_counts[names[class_id]] += 1
            image_class_set.add(names[class_id])
            split_boxes += 1

        for cls in image_class_set:
            images_per_class[cls] += 1

    total_labels += len(label_files)

    split_summary.append(
        {
            "split": split,
            "images": len(image_files),
            "labels": len(label_files),
            "missing_labels": len(split_missing_labels),
            "orphan_labels": len(split_orphan_labels),
            "boxes": split_boxes,
            "invalid_boxes": split_invalid,
        }
    )

# ------------------------------------------------------------
# Duplicate analysis
# ------------------------------------------------------------

duplicate_groups = {
    sha: files
    for sha, files in hash_to_files.items()
    if len(files) > 1
}

duplicate_files = sum(
    len(files) for files in duplicate_groups.values()
)

# ------------------------------------------------------------
# Write split summary
# ------------------------------------------------------------

with open(
    OUT / "split_summary.csv",
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "split",
            "images",
            "labels",
            "missing_labels",
            "orphan_labels",
            "boxes",
            "invalid_boxes",
        ],
    )

    writer.writeheader()
    writer.writerows(split_summary)

# ------------------------------------------------------------
# Write class distribution
# ------------------------------------------------------------

with open(
    OUT / "class_distribution.csv",
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)
    writer.writerow(
        ["class", "annotations", "images"]
    )

    for cls in names:
        writer.writerow(
            [
                cls,
                class_counts[cls],
                images_per_class[cls],
            ]
        )

# ------------------------------------------------------------
# Write orphan labels
# ------------------------------------------------------------

with open(
    OUT / "orphan_labels.csv",
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)
    writer.writerow(["split", "label_stem"])
    writer.writerows(orphan_labels)

# ------------------------------------------------------------
# Write missing labels
# ------------------------------------------------------------

with open(
    OUT / "missing_labels.csv",
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)
    writer.writerow(["split", "image_stem"])
    writer.writerows(missing_labels)

# ------------------------------------------------------------
# Write invalid boxes
# ------------------------------------------------------------

with open(
    OUT / "invalid_boxes.csv",
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)
    writer.writerow(
        ["split", "image", "reason", "details"]
    )
    writer.writerows(invalid_boxes)

# ------------------------------------------------------------
# Write invalid class IDs
# ------------------------------------------------------------

with open(
    OUT / "invalid_class_ids.csv",
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)
    writer.writerow(
        ["split", "image", "class_id", "line"]
    )
    writer.writerows(invalid_class_ids)

# ------------------------------------------------------------
# Write duplicate groups
# ------------------------------------------------------------

with open(
    OUT / "duplicate_groups.csv",
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)
    writer.writerow(
        ["sha256", "file_count", "files"]
    )

    for sha, files in sorted(
        duplicate_groups.items()
    ):
        writer.writerow(
            [sha, len(files), " | ".join(files)]
        )

# ------------------------------------------------------------
# Final report
# ------------------------------------------------------------

report = []

report.append("=" * 70)
report.append("EXTERNAL GARBAGE DATASET AUDIT")
report.append("=" * 70)
report.append("")
report.append(f"Images:              {total_images}")
report.append(f"Valid images:        {valid_images}")
report.append(f"Corrupt images:      {len(corrupt_images)}")
report.append(f"Label files:         {total_labels}")
report.append(f"Missing labels:      {len(missing_labels)}")
report.append(f"Orphan labels:       {len(orphan_labels)}")
report.append(f"Invalid boxes:       {len(invalid_boxes)}")
report.append(f"Invalid class IDs:   {len(invalid_class_ids)}")
report.append(f"Duplicate groups:    {len(duplicate_groups)}")
report.append(f"Duplicate files:     {duplicate_files}")
report.append("")
report.append("CLASSES")
report.append("-" * 70)

for cls in names:
    report.append(
        f"{cls:20s} "
        f"annotations={class_counts[cls]:6d} "
        f"images={images_per_class[cls]:6d}"
    )

report.append("")
report.append("SPLITS")
report.append("-" * 70)

for row in split_summary:
    report.append(
        f"{row['split']:6s} "
        f"images={row['images']:5d} "
        f"labels={row['labels']:5d} "
        f"boxes={row['boxes']:6d} "
        f"missing={row['missing_labels']:3d} "
        f"orphan={row['orphan_labels']:3d} "
        f"invalid={row['invalid_boxes']:3d}"
    )

report.append("")
report.append("ORPHAN LABELS")
report.append("-" * 70)

for split, stem in orphan_labels:
    report.append(f"{split}: {stem}")

report.append("")
report.append("FINAL STATUS")
report.append("-" * 70)

if (
    len(corrupt_images) == 0
    and len(missing_labels) == 0
    and len(orphan_labels) == 0
    and len(invalid_boxes) == 0
    and len(invalid_class_ids) == 0
):
    report.append("PASS — structural annotation audit is clean.")
else:
    report.append("REVIEW REQUIRED — dataset has structural issues.")

(OUT / "audit_report.txt").write_text(
    "\n".join(report),
    encoding="utf-8"
)

print("\n".join(report))
print()
print(f"Reports saved to: {OUT}")
