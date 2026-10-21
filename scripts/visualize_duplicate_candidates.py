from pathlib import Path
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(__file__).resolve().parents[1]

pairs = [
    (
        "TACO TEST - taco_0959",
        ROOT / "data/processed/taco_detection_practical/images/test/taco_0959.jpg",
        "EXTERNAL TRAIN - plastic623",
        ROOT / "data/external/garbage_object_detection/train/images/plastic623_jpg.rf.4a11123780f0d1a5bbf88fedbc624b6c.jpg",
    ),
    (
        "TACO TRAIN - taco_0978",
        ROOT / "data/processed/taco_detection_practical/images/train/taco_0978.jpg",
        "EXTERNAL TRAIN - glass1254",
        ROOT / "data/external/garbage_object_detection/train/images/glass1254_jpg.rf.3d8e3dacd31e8edc4ce54b1af8dd6def.jpg",
    ),
    (
        "TACO TRAIN - taco_0978",
        ROOT / "data/processed/taco_detection_practical/images/train/taco_0978.jpg",
        "EXTERNAL VALID - glass1334",
        ROOT / "data/external/garbage_object_detection/valid/images/glass1334_jpg.rf.948238b4cd6db6679905bee78b732e44.jpg",
    ),
]

OUTPUT = ROOT / "results/dataset_audit/taco_external_duplicate_audit"
OUTPUT.mkdir(parents=True, exist_ok=True)


def prepare(path, width=700, height=500):
    im = Image.open(path).convert("RGB")
    im.thumbnail((width, height))
    
    canvas = Image.new("RGB", (width, height), "white")
    x = (width - im.width) // 2
    y = (height - im.height) // 2
    canvas.paste(im, (x, y))
    
    return canvas


for index, (label_a, path_a, label_b, path_b) in enumerate(pairs, 1):

    left = prepare(path_a)
    right = prepare(path_b)

    canvas = Image.new("RGB", (1400, 570), "white")
    canvas.paste(left, (0, 50))
    canvas.paste(right, (700, 50))

    draw = ImageDraw.Draw(canvas)

    draw.text((10, 15), label_a, fill="black")
    draw.text((710, 15), label_b, fill="black")

    output = OUTPUT / f"candidate_{index}.jpg"
    canvas.save(output, quality=95)

    print(output)

print("\nVisual comparison images created.")