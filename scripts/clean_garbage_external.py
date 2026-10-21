from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "data" / "external" / "garbage_object_detection"
OUTPUT = ROOT / "data" / "processed" / "garbage_external_clean"

SPLITS = ["train", "valid", "test"]

for split in SPLITS:
    (OUTPUT / split / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT / split / "labels").mkdir(parents=True, exist_ok=True)

removed_boxes = 0
removed_files = 0
copied_images = 0
copied_labels = 0

for split in SPLITS:

    src_images = SOURCE / split / "images"
    src_labels = SOURCE / split / "labels"

    dst_images = OUTPUT / split / "images"
    dst_labels = OUTPUT / split / "labels"

    for image_path in src_images.iterdir():

        if not image_path.is_file():
            continue

        shutil.copy2(
            image_path,
            dst_images / image_path.name
        )

        copied_images += 1

    for label_path in src_labels.glob("*.txt"):

        output_lines = []

        with open(label_path, "r", encoding="utf-8") as f:

            for line in f:
                line = line.strip()

                if not line:
                    continue

                parts = line.split()

                if len(parts) != 5:
                    print(
                        f"WARNING: malformed line skipped: "
                        f"{split}/{label_path.name}"
                    )
                    removed_boxes += 1
                    continue

                class_id, x, y, w, h = map(float, parts)

                # Remove zero/negative-area boxes.
                if w <= 0 or h <= 0:
                    removed_boxes += 1
                    continue

                output_lines.append(line)

        # Every source image should have a label file.
        output_label = dst_labels / label_path.name

        with open(output_label, "w", encoding="utf-8") as f:
            if output_lines:
                f.write("\n".join(output_lines) + "\n")
            else:
                # Keep empty label file if all annotations were invalid.
                f.write("")

        copied_labels += 1

print("=" * 70)
print("EXTERNAL DATASET CLEANING COMPLETE")
print("=" * 70)

print(f"Images copied:          {copied_images}")
print(f"Label files copied:     {copied_labels}")
print(f"Invalid boxes removed:  {removed_boxes}")
print()

for split in SPLITS:
    image_count = len(list((OUTPUT / split / "images").iterdir()))
    label_count = len(list((OUTPUT / split / "labels").glob("*.txt")))

    print(
        f"{split:6s} "
        f"images={image_count} "
        f"labels={label_count}"
    )

print()
print("OUTPUT:")
print(OUTPUT)