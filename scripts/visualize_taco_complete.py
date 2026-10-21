from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random


ROOT = Path(__file__).resolve().parents[1]

DATASET = ROOT / "data" / "processed" / "taco_detection_complete"
OUTPUT = ROOT / "results" / "dataset_audit" / "taco" / "visual_check_complete"

SPLIT = "test"
NUM_IMAGES = 20
SEED = 42

IMAGE_DIR = DATASET / "images" / SPLIT
LABEL_DIR = DATASET / "labels" / SPLIT

OUTPUT.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Load class names
# ------------------------------------------------------------

class_names = {}

yaml_file = DATASET / "data.yaml"

for line in yaml_file.read_text(encoding="utf-8").splitlines():
    line = line.strip()

    if ":" not in line:
        continue

    left, right = line.split(":", 1)
    left = left.strip()
    right = right.strip()

    if left.isdigit():
        class_names[int(left)] = right.strip('"').strip("'")


# ------------------------------------------------------------
# Select images
# ------------------------------------------------------------

images = sorted(
    list(IMAGE_DIR.glob("*.jpg"))
    + list(IMAGE_DIR.glob("*.jpeg"))
    + list(IMAGE_DIR.glob("*.png"))
)

rng = random.Random(SEED)
rng.shuffle(images)

selected = images[:min(NUM_IMAGES, len(images))]


# ------------------------------------------------------------
# Draw
# ------------------------------------------------------------

for index, image_path in enumerate(selected, start=1):

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    if not label_path.exists():
        continue

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    W, H = image.size

    lines = label_path.read_text(
        encoding="utf-8"
    ).strip().splitlines()

    for line in lines:

        parts = line.split()

        if len(parts) != 5:
            continue

        cls_id = int(parts[0])

        xc = float(parts[1])
        yc = float(parts[2])
        bw = float(parts[3])
        bh = float(parts[4])

        x1 = (xc - bw / 2) * W
        y1 = (yc - bh / 2) * H
        x2 = (xc + bw / 2) * W
        y2 = (yc + bh / 2) * H

        name = class_names.get(
            cls_id,
            f"class_{cls_id}"
        )

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=max(2, min(W, H) // 300)
        )

        draw.text(
            (x1, max(0, y1 - 18)),
            f"{cls_id}: {name}",
            fill="red"
        )

    output_path = OUTPUT / f"{index:02d}_{image_path.name}"

    image.save(output_path, quality=95)


print("=" * 70)
print("COMPLETE 59-CLASS TACO VISUAL CHECK")
print("=" * 70)
print(f"Dataset: {DATASET}")
print(f"Split:   {SPLIT}")
print(f"Images:  {len(selected)}")
print()
print("Output:")
print(OUTPUT)
print("=" * 70)