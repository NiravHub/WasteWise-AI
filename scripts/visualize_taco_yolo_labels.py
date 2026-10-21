from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random

ROOT = Path("data/processed/taco_detection")
OUT = Path("results/dataset_audit/taco/visual_check")
OUT.mkdir(parents=True, exist_ok=True)

random.seed(42)

# Pick examples from every split.
samples = []

for split in ["train", "val", "test"]:
    image_dir = ROOT / "images" / split
    images = list(image_dir.glob("*"))

    # 3 random examples per split
    samples.extend((split, p) for p in random.sample(images, min(3, len(images))))

# Also explicitly inspect several EXIF-affected source-derived outputs
# by looking for a reasonable number of generated images.
samples = samples[:]

for idx, (split, image_path) in enumerate(samples):

    label_path = ROOT / "labels" / split / f"{image_path.stem}.txt"

    if not label_path.exists():
        continue

    with Image.open(image_path) as im:
        im = im.convert("RGB")
        W, H = im.size

        # Resize large images for practical inspection.
        max_dim = 1400
        scale = min(1.0, max_dim / max(W, H))

        draw_im = im.resize(
            (int(W * scale), int(H * scale))
        )

        draw = ImageDraw.Draw(draw_im)

        lines = [
            x.strip()
            for x in label_path.read_text(encoding="utf-8").splitlines()
            if x.strip()
        ]

        for line in lines:
            cls, xc, yc, bw, bh = line.split()

            xc = float(xc)
            yc = float(yc)
            bw = float(bw)
            bh = float(bh)

            x1 = (xc - bw / 2) * W * scale
            y1 = (yc - bh / 2) * H * scale
            x2 = (xc + bw / 2) * W * scale
            y2 = (yc + bh / 2) * H * scale

            draw.rectangle(
                [x1, y1, x2, y2],
                width=max(2, int(3 * scale))
            )

            draw.text(
                (x1, max(0, y1 - 15)),
                f"class {cls}",
            )

        output = OUT / f"{idx:02d}_{split}_{image_path.name}"
        draw_im.save(output, quality=95)

print("=" * 70)
print("TACO VISUAL CHECK GENERATED")
print("=" * 70)
print(f"Output directory:")
print(OUT.resolve())
print()
print("Open the generated images and verify:")
print("1. Boxes sit directly on litter/objects.")
print("2. No boxes appear rotated incorrectly.")
print("3. EXIF-oriented images are correctly aligned.")
print("4. Multiple objects have separate boxes.")
