from pathlib import Path
import random

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("data/processed/taco_detection_practical")
OUTPUT = Path("results/dataset_audit/taco/visual_check_practical")

CLASS_NAMES = [
    "plastic",
    "glass",
    "metal",
    "paper_cardboard",
    "biological",
    "other_trash",
]

SEED = 42
SAMPLES_PER_SPLIT = 10


def load_labels(label_path, image_width, image_height):
    labels = []

    if not label_path.exists():
        return labels

    text = label_path.read_text(encoding="utf-8").strip()

    if not text:
        return labels

    for line in text.splitlines():
        parts = line.split()

        if len(parts) != 5:
            continue

        class_id = int(parts[0])
        x, y, w, h = map(float, parts[1:])

        # YOLO normalized coordinates -> pixel coordinates
        cx = x * image_width
        cy = y * image_height
        bw = w * image_width
        bh = h * image_height

        left = int(cx - bw / 2)
        top = int(cy - bh / 2)
        right = int(cx + bw / 2)
        bottom = int(cy + bh / 2)

        labels.append(
            (
                class_id,
                left,
                top,
                right,
                bottom,
            )
        )

    return labels


def draw_image(image_path, label_path, output_path):
    image = Image.open(image_path).convert("RGB")

    labels = load_labels(
        label_path,
        image.width,
        image.height,
    )

    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except Exception:
        font = ImageFont.load_default()

    for class_id, left, top, right, bottom in labels:
        name = (
            CLASS_NAMES[class_id]
            if 0 <= class_id < len(CLASS_NAMES)
            else f"class_{class_id}"
        )

        draw.rectangle(
            [left, top, right, bottom],
            outline="red",
            width=max(3, image.width // 500),
        )

        text_bbox = draw.textbbox(
            (left, top),
            name,
            font=font,
        )

        text_left = left
        text_top = max(0, top - (text_bbox[3] - text_bbox[1]) - 6)
        text_right = text_left + (text_bbox[2] - text_bbox[0]) + 10
        text_bottom = text_top + (text_bbox[3] - text_bbox[1]) + 6

        draw.rectangle(
            [text_left, text_top, text_right, text_bottom],
            fill="red",
        )

        draw.text(
            (text_left + 5, text_top + 3),
            name,
            fill="white",
            font=font,
        )

    # Add dataset/split information
    split_name = image_path.parent.name

    info = (
        f"{split_name} | "
        f"{image_path.name} | "
        f"{len(labels)} boxes"
    )

    info_bbox = draw.textbbox(
        (0, 0),
        info,
        font=font,
    )

    info_width = info_bbox[2] - info_bbox[0]
    info_height = info_bbox[3] - info_bbox[1]

    draw.rectangle(
        [0, 0, info_width + 20, info_height + 16],
        fill="black",
    )

    draw.text(
        (10, 8),
        info,
        fill="white",
        font=font,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image.save(
        output_path,
        quality=95,
    )


def main():
    print("=" * 70)
    print("TACO PRACTICAL 6-CLASS VISUAL VERIFICATION")
    print("=" * 70)

    if not ROOT.exists():
        raise RuntimeError(
            f"Dataset directory not found: {ROOT}"
        )

    random.seed(SEED)

    if OUTPUT.exists():
        for old_file in OUTPUT.rglob("*"):
            if old_file.is_file():
                old_file.unlink()

    total_generated = 0

    for split in ["train", "val", "test"]:
        image_dir = ROOT / "images" / split
        label_dir = ROOT / "labels" / split
        output_dir = OUTPUT / split

        images = sorted(
            p
            for p in image_dir.iterdir()
            if p.suffix.lower() in {
                ".jpg",
                ".jpeg",
                ".png",
            }
        )

        if len(images) <= SAMPLES_PER_SPLIT:
            selected = images
        else:
            selected = random.sample(
                images,
                SAMPLES_PER_SPLIT,
            )

        print(
            f"{split:5}: selecting "
            f"{len(selected)} images"
        )

        for index, image_path in enumerate(selected, start=1):
            label_path = (
                label_dir /
                f"{image_path.stem}.txt"
            )

            output_path = (
                output_dir /
                f"{index:02d}_{image_path.name}"
            )

            draw_image(
                image_path,
                label_path,
                output_path,
            )

            total_generated += 1

    print()
    print("=" * 70)
    print("VISUAL VERIFICATION SET CREATED")
    print("=" * 70)
    print(f"Images generated: {total_generated}")
    print(f"Output directory: {OUTPUT.resolve()}")
    print()
    print("Open the generated images and inspect:")
    print("  - bounding boxes")
    print("  - class names")
    print("  - multi-object scenes")
    print("  - EXIF-oriented images")
    print("  - plastic / glass / metal / paper-cardboard")
    print("  - biological / other_trash")
    print("=" * 70)


if __name__ == "__main__":
    main()
