import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

ROOT = Path("data/external/taco/extracted/TACO/data")
MANIFEST = Path(
    "results/dataset_audit/taco/taco_image_manifest_corrected.csv"
)

OUT = Path("results/dataset_audit/taco/orientation_check")
OUT.mkdir(parents=True, exist_ok=True)

# Load master annotations
with open(ROOT / "annotations.json", "r", encoding="utf-8") as f:
    data = json.load(f)

images = {img["id"]: img for img in data["images"]}

annotations = {}

for ann in data["annotations"]:
    annotations.setdefault(ann["image_id"], []).append(ann)

# Load corrected manifest
manifest = {}

with open(MANIFEST, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        manifest[int(row["image_id"])] = row


def transform_bbox(bbox, W, H, orientation):
    x, y, w, h = bbox

    if orientation == 6:
        # 90 degrees clockwise
        nx = H - (y + h)
        ny = x
        nw = h
        nh = w

    elif orientation == 8:
        # 90 degrees counter-clockwise
        nx = y
        ny = W - (x + w)
        nw = h
        nh = w

    else:
        return x, y, w, h

    return nx, ny, nw, nh


# Select affected images with annotations
selected = []

for image_id, row in manifest.items():

    if row["status"] != "DIMENSION_MISMATCH":
        continue

    if image_id not in annotations:
        continue

    selected.append(image_id)

    if len(selected) >= 10:
        break


print("Selected diagnostic images:", len(selected))
print()

for image_id in selected:

    row = manifest[image_id]
    path = Path(row["image_path"])

    with Image.open(path) as im:

        orientation = im.getexif().get(274, 1)

        original_width = im.width
        original_height = im.height

        normalized = ImageOps.exif_transpose(im)

        metadata_width = int(row["width"])
        metadata_height = int(row["height"])

        print("Image:", image_id)
        print("File:", row["file_name"])
        print(
            "Metadata:",
            metadata_width,
            "x",
            metadata_height
        )
        print(
            "Stored:",
            original_width,
            "x",
            original_height
        )
        print("EXIF orientation:", orientation)
        print(
            "Normalized:",
            normalized.width,
            "x",
            normalized.height
        )

        canvas = normalized.convert("RGB")
        draw = ImageDraw.Draw(canvas)

        for ann in annotations[image_id]:

            bbox = ann["bbox"]

            x, y, w, h = transform_bbox(
                bbox,
                metadata_width,
                metadata_height,
                orientation
            )

            draw.rectangle(
                [
                    x,
                    y,
                    x + w,
                    y + h
                ],
                outline="red",
                width=max(3, int(canvas.width / 700))
            )

            draw.text(
                (x, max(0, y - 20)),
                "cat=" + str(ann["category_id"]),
                fill="red"
            )

        max_dim = 1400

        scale = min(
            1.0,
            max_dim / max(canvas.width, canvas.height)
        )

        if scale < 1:
            canvas = canvas.resize(
                (
                    int(canvas.width * scale),
                    int(canvas.height * scale)
                )
            )

        output = OUT / (
            "image_" + str(image_id) +
            "_orientation_" + str(orientation) +
            ".jpg"
        )

        canvas.save(output, quality=95)

        print("Saved:", output)
        print()

print("=" * 70)
print("Orientation diagnostic complete")
print("=" * 70)
print("Output folder:", OUT)
