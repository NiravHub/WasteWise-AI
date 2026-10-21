from pathlib import Path
import random
from PIL import Image, ImageDraw

ROOT = Path(".")
DATASET = ROOT / "data" / "processed" / "taco_detection_corrected"
OUT = ROOT / "results" / "dataset_audit" / "taco" / "yolo_visual_qa"

CLASS_NAMES = [
    "Cigarette",
    "Unlabeled litter",
    "Plastic film",
    "Clear plastic bottle",
    "Other plastic",
    "Other plastic wrapper",
    "Drink can",
    "Plastic bottle cap",
    "Plastic straw",
    "Broken glass",
    "Styrofoam piece",
    "Glass bottle",
    "Disposable plastic cup",
    "Pop tab",
    "Other carton",
    "Normal paper",
    "Metal bottle cap",
    "Plastic lid",
    "Paper cup",
    "Corrugated carton",
    "Aluminium foil",
    "Single-use carrier bag",
    "Other plastic bottle",
]

OUT.mkdir(parents=True, exist_ok=True)

random.seed(42)

for split in ["train", "val", "test"]:

    image_dir = DATASET / split / "images"
    label_dir = DATASET / split / "labels"

    images = list(image_dir.glob("*.jpg"))

    # Five random images from each split.
    selected = random.sample(images, min(5, len(images)))

    print()
    print("=" * 70)
    print(f"{split.upper()} VISUAL QA")
    print("=" * 70)

    for image_path in selected:

        label_path = label_dir / f"{image_path.stem}.txt"

        with Image.open(image_path) as im:
            image = im.convert("RGB")

        W, H = image.size

        draw = ImageDraw.Draw(image)

        with open(label_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        for line in lines:

            parts = line.split()

            if len(parts) != 5:
                raise RuntimeError(
                    f"Invalid YOLO line in {label_path}: {line}"
                )

            cls = int(parts[0])
            xc = float(parts[1])
            yc = float(parts[2])
            bw = float(parts[3])
            bh = float(parts[4])

            if not (0 <= xc <= 1 and 0 <= yc <= 1):
                raise RuntimeError(
                    f"Invalid center coordinates: {line}"
                )

            if not (0 < bw <= 1 and 0 < bh <= 1):
                raise RuntimeError(
                    f"Invalid box dimensions: {line}"
                )

            x1 = (xc - bw / 2) * W
            y1 = (yc - bh / 2) * H
            x2 = (xc + bw / 2) * W
            y2 = (yc + bh / 2) * H

            draw.rectangle(
                [x1, y1, x2, y2],
                outline="red",
                width=4
            )

            label = (
                CLASS_NAMES[cls]
                if 0 <= cls < len(CLASS_NAMES)
                else f"class_{cls}"
            )

            draw.text(
                (x1 + 3, max(0, y1 - 18)),
                label,
                fill="red"
            )

        output = OUT / f"{split}_{image_path.stem}.jpg"

        image.save(output, quality=95)

        print(f"{split}: {image_path.name} -> {output}")

print()
print("=" * 70)
print("YOLO VISUAL QA COMPLETE")
print("=" * 70)
print(f"Output: {OUT}")
