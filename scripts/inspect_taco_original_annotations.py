import json
from pathlib import Path

ROOT = Path.cwd()

TACO = ROOT / "data/external/taco/extracted/TACO/data/annotations.json"
LABEL_ROOT = ROOT / "data/processed/taco_detection/labels/test"

with open(TACO, "r", encoding="utf-8") as f:
    data = json.load(f)

images = {img["id"]: img for img in data["images"]}
categories = {cat["id"]: cat["name"] for cat in data["categories"]}

for image_id in [281, 1498]:

    print()
    print("=" * 80)
    print(f"IMAGE ID: {image_id}")
    print("=" * 80)

    if image_id not in images:
        print("IMAGE ID NOT FOUND")
        continue

    info = images[image_id]

    print("TACO file:", info["file_name"])
    print("Size:", info["width"], "x", info["height"])

    anns = [
        ann for ann in data["annotations"]
        if ann["image_id"] == image_id
    ]

    print()
    print("ALL ORIGINAL TACO ANNOTATIONS")
    print("-" * 80)

    for ann in anns:
        name = categories.get(ann["category_id"], "UNKNOWN")
        print(
            f"category_id={ann['category_id']:>2} | "
            f"{name:<35} | "
            f"bbox={ann['bbox']}"
        )

    print()
    print("SELECTED FOR OUR 23-CLASS DETECTOR")
    print("-" * 80)

    selected = {
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
    }

    selected_anns = []

    for ann in anns:
        name = categories.get(ann["category_id"], "UNKNOWN")

        if name in selected:
            selected_anns.append(ann)
            print(
                f"{name:<35} | bbox={ann['bbox']}"
            )

    print()
    print("GENERATED YOLO LABEL")
    print("-" * 80)

    # Our builder names images taco_XXXX.jpg, so inspect corresponding label.
    possible = [
        LABEL_ROOT / f"taco_{image_id:04d}.txt",
        LABEL_ROOT / f"{image_id}.txt",
    ]

    found = False

    for label_path in possible:
        if label_path.exists():
            print(label_path)
            print(label_path.read_text(encoding="utf-8"))
            found = True
            break

    if not found:
        print("Could not locate generated label using expected names.")

