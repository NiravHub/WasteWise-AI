from pathlib import Path
import json
import csv
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(".")
TACO = ROOT / "data/external/taco/extracted/TACO"
MASTER = TACO / "data/annotations.json"
MANIFEST = ROOT / "results/dataset_audit/taco/taco_image_manifest_corrected.csv"
OUT = ROOT / "results/dataset_audit/taco/orientation_check_corrected"

OUT.mkdir(parents=True, exist_ok=True)

with open(MASTER, "r", encoding="utf-8") as f:
    data = json.load(f)

images = {img["id"]: img for img in data["images"]}

# Exact corrected image paths
manifest = {}
with open(MANIFEST, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        manifest[int(row["image_id"])] = row

# First 10 images affected by EXIF orientation
selected = []

for image_id, img in images.items():
    row = manifest.get(image_id)
    if not row:
        continue

    path = Path(row["image_path"])
    if not path.exists():
        continue

    with Image.open(path) as im:
        exif = im.getexif()
        orientation = exif.get(274, 1)

    if orientation in (6, 8):
        selected.append(image_id)

    if len(selected) >= 10:
        break

annotations_by_image = {}

for ann in data["annotations"]:
    annotations_by_image.setdefault(ann["image_id"], []).append(ann)

print("Selected images:", selected)

for image_id in selected:
    img_info = images[image_id]
    row = manifest[image_id]

    path = Path(row["image_path"])

    with Image.open(path) as im:
        orientation = im.getexif().get(274, 1)

        # Correct the IMAGE orientation only.
        im = ImageOps.exif_transpose(im).convert("RGB")

        W, H = im.size

        draw = ImageDraw.Draw(im)

        for ann in annotations_by_image.get(image_id, []):
            x = float(ann["bbox"][0])
            y = float(ann["bbox"][1])
            w = float(ann["bbox"][2])
            h = float(ann["bbox"][3])

            # IMPORTANT:
            # TACO bbox is already expressed in the
            # display-oriented coordinate system.
            # DO NOT rotate/transform the bbox here.

            x1 = max(0, min(W, x))
            y1 = max(0, min(H, y))
            x2 = max(0, min(W, x + w))
            y2 = max(0, min(H, y + h))

            if x2 > x1 and y2 > y1:
                draw.rectangle(
                    [x1, y1, x2, y2],
                    outline="red",
                    width=3
                )

        output = OUT / f"image_{image_id}_orientation_{orientation}_corrected.jpg"
        im.save(output, quality=95)

        print(
            f"{image_id}: EXIF={orientation}, "
            f"size={W}x{H}, output={output}"
        )

print()
print("DONE")
print("Output:", OUT)
