from pathlib import Path
import json
import csv
from collections import Counter, defaultdict
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

MASTER_JSON = ROOT / "data" / "external" / "taco" / "extracted" / "TACO" / "data" / "annotations.json"
MANIFEST = ROOT / "results" / "dataset_audit" / "taco" / "taco_image_manifest_corrected.csv"

SELECTED = {
    59: "Cigarette",
    58: "Unlabeled litter",
    36: "Plastic film",
    5: "Clear plastic bottle",
    29: "Other plastic",
    39: "Other plastic wrapper",
    12: "Drink can",
    7: "Plastic bottle cap",
    55: "Plastic straw",
    9: "Broken glass",
    57: "Styrofoam piece",
    6: "Glass bottle",
    21: "Disposable plastic cup",
    50: "Pop tab",
    14: "Other carton",
    33: "Normal paper",
    8: "Metal bottle cap",
    27: "Plastic lid",
    20: "Paper cup",
    17: "Corrugated carton",
    0: "Aluminium foil",
    40: "Single-use carrier bag",
    4: "Other plastic bottle",
}


def transform_bbox(x, y, w, h, W, H, orientation):

    if orientation == 3:
        return W - (x + w), H - (y + h), w, h

    if orientation == 6:
        return W - (y + h), x, h, w

    if orientation == 8:
        return y, H - (x + w), h, w

    return x, y, w, h


def main():

    with MASTER_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)

    manifest = {}

    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            manifest[int(row["image_id"])] = {
                "path": Path(row["image_path"]),
                "file_name": row["file_name"],
                "W": int(row["width"]),
                "H": int(row["height"]),
            }

    images = {
        int(x["id"]): x
        for x in data["images"]
    }

    invalid = []
    reasons = Counter()
    by_class = Counter()
    by_orientation = Counter()
    by_image = Counter()

    selected_annotations = 0

    for ann in data["annotations"]:

        cat = int(ann["category_id"])

        if cat not in SELECTED:
            continue

        selected_annotations += 1

        image_id = int(ann["image_id"])

        info = manifest[image_id]
        image_meta = images[image_id]

        W = info["W"]
        H = info["H"]

        path = info["path"]

        try:
            with Image.open(path) as im:
                orientation = int(im.getexif().get(274, 1))
        except Exception:
            orientation = -1

        x, y, w, h = map(float, ann["bbox"])

        nx, ny, nw, nh = transform_bbox(
            x, y, w, h, W, H, orientation
        )

        reason = None

        if w <= 0 or h <= 0:
            reason = "original_nonpositive_size"

        elif nx < 0:
            reason = "negative_x"

        elif ny < 0:
            reason = "negative_y"

        elif nw <= 0:
            reason = "nonpositive_width"

        elif nh <= 0:
            reason = "nonpositive_height"

        elif nx + nw > W:
            reason = "right_overflow"

        elif ny + nh > H:
            reason = "bottom_overflow"

        if reason:

            invalid.append({
                "annotation_id": ann["id"],
                "image_id": image_id,
                "class_id": cat,
                "class_name": SELECTED[cat],
                "orientation": orientation,
                "file_name": info["file_name"],
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "new_x": nx,
                "new_y": ny,
                "new_w": nw,
                "new_h": nh,
                "W": W,
                "H": H,
                "reason": reason,
            })

            reasons[reason] += 1
            by_class[SELECTED[cat]] += 1
            by_orientation[orientation] += 1
            by_image[image_id] += 1

    print("=" * 75)
    print("TACO INVALID-BOX DIAGNOSTIC")
    print("=" * 75)

    print()
    print(f"Selected annotations: {selected_annotations}")
    print(f"Invalid annotations:  {len(invalid)}")

    print()
    print("INVALIDITY REASONS")
    print("-" * 50)

    for reason, count in reasons.most_common():
        print(f"{reason:30s} {count}")

    print()
    print("INVALID BOXES BY EXIF ORIENTATION")
    print("-" * 50)

    for orientation, count in sorted(by_orientation.items()):
        print(f"orientation {orientation}: {count}")

    print()
    print("INVALID BOXES BY CLASS")
    print("-" * 50)

    for name, count in by_class.most_common():
        print(f"{name:30s} {count}")

    print()
    print("TOP AFFECTED IMAGES")
    print("-" * 50)

    for image_id, count in by_image.most_common(20):

        info = manifest[image_id]

        print(
            f"image {image_id:4d} | "
            f"{count:2d} invalid | "
            f"{info['file_name']}"
        )

    out = ROOT / "results" / "dataset_audit" / "taco" / "invalid_bbox_diagnostic.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8", newline="") as f:

        fields = [
            "annotation_id",
            "image_id",
            "class_id",
            "class_name",
            "orientation",
            "file_name",
            "x",
            "y",
            "w",
            "h",
            "new_x",
            "new_y",
            "new_w",
            "new_h",
            "W",
            "H",
            "reason",
        ]

        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(invalid)

    print()
    print(f"Detailed CSV:")
    print(out)


if __name__ == "__main__":
    main()
