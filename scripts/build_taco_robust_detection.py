import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]

TACO_ROOT = ROOT / "data" / "external" / "taco" / "extracted" / "TACO"
ANNOTATIONS = TACO_ROOT / "data" / "annotations.json"

OUTPUT = ROOT / "data" / "processed" / "taco_detection_robust"

SEED = 42

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10


# ============================================================
# VALIDATED 59 -> 33 CLASS TAXONOMY
# ============================================================

TAXONOMY = {
    "cigarette": ["Cigarette"],
    "unlabeled_litter": ["Unlabeled litter"],

    "plastic_film_wrapper": [
        "Plastic film",
        "Other plastic wrapper",
        "Crisp packet",
    ],

    "plastic_bottle": [
        "Clear plastic bottle",
        "Other plastic bottle",
    ],

    "other_plastic": ["Other plastic"],

    "bottle_cap": [
        "Plastic bottle cap",
        "Metal bottle cap",
    ],

    "plastic_straw": [
        "Plastic straw",
        "Paper straw",
    ],

    "glass_bottle": ["Glass bottle"],
    "broken_glass": ["Broken glass"],

    "glass_container": [
        "Glass jar",
        "Glass cup",
    ],

    "plastic_cup": [
        "Disposable plastic cup",
        "Foam cup",
        "Other plastic cup",
        "Paper cup",
    ],

    "paper": [
        "Normal paper",
        "Magazine paper",
        "Wrapping paper",
        "Tissues",
    ],

    "carton": [
        "Other carton",
        "Drink carton",
        "Corrugated carton",
        "Meal carton",
        "Egg carton",
        "Pizza box",
    ],

    "metal_can": [
        "Drink can",
        "Food Can",
    ],

    "metal_misc": [
        "Aerosol",
        "Scrap metal",
    ],

    "pop_tab": ["Pop tab"],

    "plastic_lid": [
        "Plastic lid",
        "Metal lid",
    ],

    "plastic_bag": [
        "Garbage bag",
        "Single-use carrier bag",
        "Polypropylene bag",
    ],

    "plastic_container": [
        "Disposable food container",
        "Foam food container",
        "Other plastic container",
        "Tupperware",
    ],

    "styrofoam": ["Styrofoam piece"],

    "plastic_utensil": ["Plastic utensils"],

    "rope_string": ["Rope & strings"],

    "paper_bag": ["Paper bag"],

    "aluminium_foil": ["Aluminium foil"],

    "six_pack_ring": ["Six pack rings"],

    "medical_packaging": [
        "Aluminium blister pack",
        "Carded blister pack",
    ],

    "food_waste": ["Food waste"],
    "shoe": ["Shoe"],
    "squeezable_tube": ["Squeezable tube"],
    "toilet_tube": ["Toilet tube"],
    "spread_tub": ["Spread tub"],
    "plastic_gloves": ["Plastic glooves"],
    "battery": ["Battery"],
}


# ============================================================
# OUTPUT CLASS ORDER
# ============================================================

CLASS_NAMES = list(TAXONOMY.keys())

CLASS_TO_ID = {
    name: idx
    for idx, name in enumerate(CLASS_NAMES)
}

SOURCE_TO_ROBUST = {}

for robust_name, source_names in TAXONOMY.items():
    for source_name in source_names:
        SOURCE_TO_ROBUST[source_name] = robust_name


# ============================================================
# LOAD TACO MASTER JSON
# ============================================================

with open(ANNOTATIONS, "r", encoding="utf-8") as f:
    taco = json.load(f)

categories = {
    cat["id"]: cat["name"]
    for cat in taco["categories"]
}

images = taco["images"]
annotations = taco["annotations"]


# ============================================================
# VALIDATE SOURCE TAXONOMY BEFORE BUILD
# ============================================================

source_class_names = set(categories.values())

# TACO contains one category definition with zero annotations:
# "Plastified paper bag".
# It is intentionally excluded because it contributes no
# training instances.

annotated_category_ids = {
    ann["category_id"]
    for ann in annotations
}

annotated_source_class_names = {
    categories[category_id]
    for category_id in annotated_category_ids
}

missing_source_classes = sorted(
    annotated_source_class_names - set(SOURCE_TO_ROBUST)
)
unknown_source_classes = sorted(
    set(SOURCE_TO_ROBUST) - source_class_names
)

if missing_source_classes:
    raise RuntimeError(
        "Unmapped TACO classes:\n"
        + "\n".join(missing_source_classes)
    )

if unknown_source_classes:
    raise RuntimeError(
        "Unknown taxonomy classes:\n"
        + "\n".join(unknown_source_classes)
    )


# ============================================================
# IMAGE PATH RESOLUTION
# ============================================================

DATA_ROOT = TACO_ROOT / "data"


def resolve_image(file_name: str) -> Path:
    """
    Master TACO annotations use paths such as:
        batch_1/000006.jpg

    Resolve directly relative to TACO/data.
    """

    path = DATA_ROOT / file_name

    if path.exists():
        return path

    # Case-insensitive fallback for extension/case differences.
    parent = path.parent
    target = path.name.lower()

    if parent.exists():
        for candidate in parent.iterdir():
            if candidate.name.lower() == target:
                return candidate

    raise FileNotFoundError(
        f"Could not resolve TACO image: {file_name}"
    )


# ============================================================
# GROUP ANNOTATIONS BY IMAGE
# ============================================================

annotations_by_image = defaultdict(list)

for ann in annotations:
    annotations_by_image[ann["image_id"]].append(ann)


# ============================================================
# VERIFY EVERY ANNOTATION CAN BE MAPPED
# ============================================================

for ann in annotations:

    category_name = categories[ann["category_id"]]

    if category_name not in SOURCE_TO_ROBUST:
        raise RuntimeError(
            f"Annotation has unmapped category: {category_name}"
        )


# ============================================================
# IMAGE-LEVEL SPLIT
# ============================================================

image_ids = [
    image["id"]
    for image in images
    if image["id"] in annotations_by_image
]

random.seed(SEED)
random.shuffle(image_ids)

n = len(image_ids)

train_end = int(n * TRAIN_RATIO)
val_end = train_end + int(n * VAL_RATIO)

train_ids = image_ids[:train_end]
val_ids = image_ids[train_end:val_end]
test_ids = image_ids[val_end:]

splits = {
    "train": train_ids,
    "val": val_ids,
    "test": test_ids,
}


# ============================================================
# CLEAN OUTPUT DIRECTORY
# ============================================================

if OUTPUT.exists():
    shutil.rmtree(OUTPUT)

for split in splits:
    (OUTPUT / "images" / split).mkdir(
        parents=True,
        exist_ok=True
    )

    (OUTPUT / "labels" / split).mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# PROCESS IMAGES + LABELS
# ============================================================

processed_images = 0
converted_boxes = 0
invalid_boxes = 0
missing_images = 0

class_counts = Counter()
split_class_counts = {
    "train": Counter(),
    "val": Counter(),
    "test": Counter(),
}

image_manifest = []

image_lookup = {
    image["id"]: image
    for image in images
}


for split, split_ids in splits.items():

    for image_id in split_ids:

        image_info = image_lookup[image_id]

        file_name = image_info["file_name"]
        image_path = resolve_image(file_name)

        try:
            with Image.open(image_path) as im:
                # Normalize EXIF orientation.
                normalized = ImageOps.exif_transpose(im)

                actual_w, actual_h = normalized.size

                output_name = (
                    f"taco_{image_id:04d}"
                    ".jpg"
                )

                output_image = (
                    OUTPUT
                    / "images"
                    / split
                    / output_name
                )

                normalized = normalized.convert("RGB")
                normalized.save(
                    output_image,
                    quality=95
                )

        except Exception as exc:
            missing_images += 1
            print(
                f"ERROR processing image "
                f"{file_name}: {exc}"
            )
            continue

        label_lines = []

        for ann in annotations_by_image[image_id]:

            category_name = categories[
                ann["category_id"]
            ]

            robust_name = SOURCE_TO_ROBUST[
                category_name
            ]

            robust_id = CLASS_TO_ID[
                robust_name
            ]

            x, y, w, h = ann["bbox"]

            # TACO bbox coordinates are already expressed
            # in the display-oriented coordinate system.
            # Do NOT rotate bbox coordinates after EXIF transpose.

            left = max(0.0, float(x))
            top = max(0.0, float(y))
            right = min(
                float(actual_w),
                float(x) + float(w)
            )
            bottom = min(
                float(actual_h),
                float(y) + float(h)
            )

            clipped_w = right - left
            clipped_h = bottom - top

            if clipped_w <= 0 or clipped_h <= 0:
                invalid_boxes += 1
                continue

            # YOLO center format.
            cx = (left + right) / 2.0 / actual_w
            cy = (top + bottom) / 2.0 / actual_h

            nw = clipped_w / actual_w
            nh = clipped_h / actual_h

            # Final numeric validation.
            if not (
                0 <= cx <= 1
                and 0 <= cy <= 1
                and 0 < nw <= 1
                and 0 < nh <= 1
            ):
                invalid_boxes += 1
                continue

            label_lines.append(
                f"{robust_id} "
                f"{cx:.6f} "
                f"{cy:.6f} "
                f"{nw:.6f} "
                f"{nh:.6f}"
            )

            converted_boxes += 1
            class_counts[robust_name] += 1
            split_class_counts[split][robust_name] += 1

        label_path = (
            OUTPUT
            / "labels"
            / split
            / f"taco_{image_id:04d}.txt"
        )

        label_path.write_text(
            "\n".join(label_lines) + "\n",
            encoding="utf-8"
        )

        processed_images += 1

        image_manifest.append({
            "image_id": image_id,
            "split": split,
            "source_file": file_name,
            "output_image": str(
                Path("images") / split / output_name
            ),
            "output_label": str(
                Path("labels")
                / split
                / f"taco_{image_id:04d}.txt"
            ),
            "width": actual_w,
            "height": actual_h,
            "annotation_count": len(
                label_lines
            ),
        })


# ============================================================
# WRITE DATASET YAML
# ============================================================

yaml_lines = [
    f"path: {OUTPUT.as_posix()}",
    "train: images/train",
    "val: images/val",
    "test: images/test",
    "names:",
]

for idx, name in enumerate(CLASS_NAMES):
    yaml_lines.append(
        f"  {idx}: {name}"
    )

(OUTPUT / "data.yaml").write_text(
    "\n".join(yaml_lines) + "\n",
    encoding="utf-8"
)


# ============================================================
# WRITE CLASS MANIFEST
# ============================================================

with open(
    OUTPUT / "class_manifest.csv",
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "robust_id",
        "class_name",
        "annotation_count",
    ])

    for idx, name in enumerate(CLASS_NAMES):
        writer.writerow([
            idx,
            name,
            class_counts[name],
        ])


# ============================================================
# WRITE IMAGE MANIFEST
# ============================================================

with open(
    OUTPUT / "image_manifest.csv",
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "image_id",
            "split",
            "source_file",
            "output_image",
            "output_label",
            "width",
            "height",
            "annotation_count",
        ]
    )

    writer.writeheader()
    writer.writerows(image_manifest)


# ============================================================
# WRITE TAXONOMY MAP
# ============================================================

with open(
    OUTPUT / "taxonomy_mapping.csv",
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "source_taco_class",
        "robust_class",
        "robust_id",
    ])

    for source_name, robust_name in sorted(
        SOURCE_TO_ROBUST.items()
    ):
        writer.writerow([
            source_name,
            robust_name,
            CLASS_TO_ID[robust_name],
        ])


# ============================================================
# WRITE SUMMARY JSON
# ============================================================

summary = {
    "source_classes": len(source_class_names),
    "robust_classes": len(CLASS_NAMES),
    "source_annotations": len(annotations),
    "converted_boxes": converted_boxes,
    "invalid_boxes": invalid_boxes,
    "processed_images": processed_images,
    "missing_images": missing_images,
    "train_images": len(train_ids),
    "val_images": len(val_ids),
    "test_images": len(test_ids),
    "annotation_retention_percent": (
        converted_boxes / len(annotations) * 100
        if annotations
        else 0
    ),
    "seed": SEED,
    "class_counts": dict(class_counts),
    "split_class_counts": {
        split: dict(counts)
        for split, counts in split_class_counts.items()
    },
}


with open(
    OUTPUT / "build_summary.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        summary,
        f,
        indent=2
    )


# ============================================================
# FINAL CONSOLE REPORT
# ============================================================

print("=" * 80)
print("TACO ROBUST 33-CLASS DATASET BUILD COMPLETE")
print("=" * 80)

print(f"Source categories:     {len(source_class_names)}")
print(f"Annotated classes:     {len(annotated_source_class_names)}")
print(f"Robust classes:        {len(CLASS_NAMES)}")
print(f"Source images:         {len(images)}")
print(f"Processed images:      {processed_images}")
print(f"Source annotations:    {len(annotations)}")
print(f"Converted boxes:       {converted_boxes}")
print(f"Invalid boxes:         {invalid_boxes}")
print(f"Missing images:        {missing_images}")

print("")
print("SPLITS")

for split in ["train", "val", "test"]:
    print(
        f"{split:5s}: "
        f"{len(splits[split]):4d} images, "
        f"{sum(split_class_counts[split].values()):4d} boxes"
    )

print("")
print("ANNOTATION RETENTION:")
print(
    f"{converted_boxes / len(annotations) * 100:.2f}%"
)

print("")
print("OUTPUT:")
print(OUTPUT)

print("")
print("IMPORTANT:")
print("The complete 59-class dataset was NOT modified.")
print("=" * 80)