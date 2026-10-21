from pathlib import Path
import json
import random
import shutil
from collections import Counter, defaultdict

from PIL import Image, ImageOps


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TACO_ROOT = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "taco"
    / "extracted"
    / "TACO"
)

ANNOTATIONS_FILE = TACO_ROOT / "data" / "annotations.json"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "taco_detection_practical"
)

SEED = 42


# ============================================================
# PRACTICAL 6-CLASS TAXONOMY
# ============================================================

PRACTICAL_CLASSES = [
    "plastic",
    "glass",
    "metal",
    "paper_cardboard",
    "biological",
    "other_trash",
]


# ============================================================
# TACO -> PRACTICAL CLASS MAPPING
# ============================================================

SOURCE_TO_PRACTICAL = {

    # --------------------------------------------------------
    # PLASTIC
    # --------------------------------------------------------

    "Clear plastic bottle": "plastic",
    "Crisp packet": "plastic",
    "Disposable food container": "plastic",
    "Disposable plastic cup": "plastic",
    "Foam cup": "plastic",
    "Foam food container": "plastic",
    "Garbage bag": "plastic",
    "Metal bottle cap": "plastic",
    "Metal lid": "plastic",
    "Other plastic": "plastic",
    "Other plastic bottle": "plastic",
    "Other plastic container": "plastic",
    "Other plastic cup": "plastic",
    "Other plastic wrapper": "plastic",
    "Paper straw": "plastic",
    "Plastic bottle cap": "plastic",
    "Plastic film": "plastic",
    "Plastic glooves": "plastic",
    "Plastic lid": "plastic",
    "Plastic straw": "plastic",
    "Plastic utensils": "plastic",
    "Polypropylene bag": "plastic",
    "Single-use carrier bag": "plastic",
    "Spread tub": "plastic",
    "Styrofoam piece": "plastic",
    "Tupperware": "plastic",

    # --------------------------------------------------------
    # GLASS
    # --------------------------------------------------------

    "Broken glass": "glass",
    "Glass bottle": "glass",
    "Glass cup": "glass",
    "Glass jar": "glass",

    # --------------------------------------------------------
    # METAL
    # --------------------------------------------------------

    "Aerosol": "metal",
    "Aluminium foil": "metal",
    "Drink can": "metal",
    "Food Can": "metal",
    "Pop tab": "metal",
    "Scrap metal": "metal",
    "Six pack rings": "metal",

    # --------------------------------------------------------
    # PAPER / CARDBOARD
    # --------------------------------------------------------

    "Corrugated carton": "paper_cardboard",
    "Drink carton": "paper_cardboard",
    "Egg carton": "paper_cardboard",
    "Magazine paper": "paper_cardboard",
    "Meal carton": "paper_cardboard",
    "Normal paper": "paper_cardboard",
    "Other carton": "paper_cardboard",
    "Paper bag": "paper_cardboard",
    "Paper cup": "paper_cardboard",
    "Pizza box": "paper_cardboard",
    "Tissues": "paper_cardboard",
    "Wrapping paper": "paper_cardboard",

    # --------------------------------------------------------
    # BIOLOGICAL
    # --------------------------------------------------------

    "Food waste": "biological",

    # --------------------------------------------------------
    # OTHER TRASH
    # --------------------------------------------------------

    "Aluminium blister pack": "other_trash",
    "Carded blister pack": "other_trash",
    "Cigarette": "other_trash",
    "Rope & strings": "other_trash",
    "Shoe": "other_trash",
    "Squeezable tube": "other_trash",
    "Toilet tube": "other_trash",
    "Unlabeled litter": "other_trash",
}


# Battery is deliberately NOT mapped.
# TACO contains only 2 Battery annotations.


# ============================================================
# OUTPUT PREPARATION
# ============================================================

def clean_output():
    """Remove previous practical dataset and recreate structure."""

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    for split in ["train", "val", "test"]:
        (
            OUTPUT_DIR
            / "images"
            / split
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            OUTPUT_DIR
            / "labels"
            / split
        ).mkdir(
            parents=True,
            exist_ok=True,
        )


# ============================================================
# IMAGE RESOLUTION
# ============================================================

def resolve_image_path(file_name):
    """
    Resolve TACO master annotation image path.

    Example:
        batch_1/000006.jpg
    """

    path = TACO_ROOT / "data" / file_name

    if path.exists():
        return path

    # Case-insensitive fallback.
    parent = path.parent
    target_name = path.name.lower()

    if parent.exists():
        for candidate in parent.iterdir():
            if candidate.name.lower() == target_name:
                return candidate

    return None


# ============================================================
# EXIF NORMALIZATION
# ============================================================

def load_rgb_normalized(path):
    """
    Normalize image orientation using EXIF.

    IMPORTANT:
    TACO bounding boxes are used AS-IS.

    We do NOT rotate/transform bbox coordinates after
    ImageOps.exif_transpose().
    """

    with Image.open(path) as im:
        normalized = ImageOps.exif_transpose(im).convert("RGB")
        return normalized.copy()


# ============================================================
# BOUNDING BOX CLIPPING
# ============================================================

def clip_bbox(
    x,
    y,
    w,
    h,
    image_width,
    image_height,
):
    """
    Clip bbox to normalized image boundaries.
    """

    left = max(0.0, x)
    top = max(0.0, y)

    right = min(
        float(image_width),
        x + w,
    )

    bottom = min(
        float(image_height),
        y + h,
    )

    new_w = right - left
    new_h = bottom - top

    if new_w <= 0 or new_h <= 0:
        return None

    return (
        left,
        top,
        new_w,
        new_h,
    )


# ============================================================
# YOLO CONVERSION
# ============================================================

def bbox_to_yolo(
    x,
    y,
    w,
    h,
    image_width,
    image_height,
):
    """
    Convert TACO xywh bbox to YOLO normalized format.
    """

    clipped = clip_bbox(
        x,
        y,
        w,
        h,
        image_width,
        image_height,
    )

    if clipped is None:
        return None

    x, y, w, h = clipped

    x_center = (
        x + w / 2.0
    ) / image_width

    y_center = (
        y + h / 2.0
    ) / image_height

    width = (
        w / image_width
    )

    height = (
        h / image_height
    )

    values = [
        x_center,
        y_center,
        width,
        height,
    ]

    if not all(
        0.0 <= value <= 1.0
        for value in values
    ):
        return None

    if width <= 0 or height <= 0:
        return None

    return values


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TACO PRACTICAL 6-CLASS DATASET BUILD")
    print("=" * 70)

    # --------------------------------------------------------
    # Check source
    # --------------------------------------------------------

    if not ANNOTATIONS_FILE.exists():
        raise FileNotFoundError(
            f"Missing TACO annotations:\n"
            f"{ANNOTATIONS_FILE}"
        )

    # --------------------------------------------------------
    # Load TACO annotations
    # --------------------------------------------------------

    with open(
        ANNOTATIONS_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    categories = {
        category["id"]: category["name"]
        for category in data["categories"]
    }

    images = data["images"]
    annotations = data["annotations"]

    image_by_id = {
        image["id"]: image
        for image in images
    }

    # --------------------------------------------------------
    # TAXONOMY VALIDATION
    # --------------------------------------------------------

    annotated_categories = {
        categories[ann["category_id"]]
        for ann in annotations
    }

    # Battery is intentionally excluded.
    excluded_categories = {
        "Battery"
    }

    unmapped = sorted(
        annotated_categories
        - set(SOURCE_TO_PRACTICAL)
        - excluded_categories
    )

    if unmapped:

        print()
        print(
            "ERROR - unmapped annotated categories:"
        )

        for name in unmapped:
            print(
                f"  {name}"
            )

        raise RuntimeError(
            "Taxonomy validation failed. "
            "Every annotated category must either "
            "be mapped or explicitly excluded."
        )

    print()
    print(
        "Intentionally excluded categories: "
        + ", ".join(
            sorted(excluded_categories)
        )
    )

    # --------------------------------------------------------
    # GROUP ANNOTATIONS BY IMAGE
    # --------------------------------------------------------

    annotations_by_image = defaultdict(list)

    for ann in annotations:

        source_name = categories[
            ann["category_id"]
        ]

        # Skip intentionally excluded classes.
        if source_name in excluded_categories:
            continue

        practical_class = SOURCE_TO_PRACTICAL[
            source_name
        ]

        annotations_by_image[
            ann["image_id"]
        ].append(
            {
                "annotation": ann,
                "source_name": source_name,
                "practical_class": practical_class,
            }
        )

    selected_image_ids = sorted(
        annotations_by_image
    )

    selected_annotations = sum(
        len(items)
        for items in annotations_by_image.values()
    )

    print()
    print(
        f"Source images:        "
        f"{len(images)}"
    )

    print(
        f"Source annotations:   "
        f"{len(annotations)}"
    )

    print(
        f"Selected images:      "
        f"{len(selected_image_ids)}"
    )

    print(
        f"Selected annotations: "
        f"{selected_annotations}"
    )

    # --------------------------------------------------------
    # DETERMINISTIC IMAGE-LEVEL SPLIT
    # --------------------------------------------------------

    rng = random.Random(SEED)

    shuffled_ids = selected_image_ids.copy()

    rng.shuffle(
        shuffled_ids
    )

    n = len(shuffled_ids)

    train_end = int(
        n * 0.80
    )

    val_end = (
        train_end
        + int(n * 0.10)
    )

    split_ids = {
        "train": shuffled_ids[
            :train_end
        ],
        "val": shuffled_ids[
            train_end:val_end
        ],
        "test": shuffled_ids[
            val_end:
        ],
    }

    print()
    print("SPLITS")
    print("-" * 70)

    for split, ids in split_ids.items():

        print(
            f"{split:5}: "
            f"{len(ids)} images"
        )

    # --------------------------------------------------------
    # CREATE OUTPUT
    # --------------------------------------------------------

    clean_output()

    class_to_id = {
        name: idx
        for idx, name in enumerate(
            PRACTICAL_CLASSES
        )
    }

    class_counts = Counter()

    processed_images = 0
    converted_boxes = 0
    invalid_boxes = 0
    missing_images = 0

    orientation_counts = Counter()

    # --------------------------------------------------------
    # PROCESS DATA
    # --------------------------------------------------------

    for split, image_ids in split_ids.items():

        for image_id in image_ids:

            image_info = image_by_id[
                image_id
            ]

            file_name = image_info[
                "file_name"
            ]

            source_path = resolve_image_path(
                file_name
            )

            if source_path is None:

                missing_images += 1

                print(
                    "WARNING: missing image: "
                    f"{file_name}"
                )

                continue

            # ------------------------------------------------
            # Load + EXIF normalize
            # ------------------------------------------------

            try:

                with Image.open(
                    source_path
                ) as original:

                    orientation = (
                        original
                        .getexif()
                        .get(
                            274,
                            1,
                        )
                    )

                orientation_counts[
                    orientation
                ] += 1

                image = load_rgb_normalized(
                    source_path
                )

            except Exception as exc:

                print(
                    "WARNING: failed image "
                    f"{file_name}: {exc}"
                )

                missing_images += 1

                continue

            width, height = image.size

            # ------------------------------------------------
            # Output names
            # ------------------------------------------------

            output_stem = (
                f"taco_{int(image_id):04d}"
            )

            output_image = (
                OUTPUT_DIR
                / "images"
                / split
                / f"{output_stem}.jpg"
            )

            output_label = (
                OUTPUT_DIR
                / "labels"
                / split
                / f"{output_stem}.txt"
            )

            # ------------------------------------------------
            # Save normalized image
            # ------------------------------------------------

            image.save(
                output_image,
                format="JPEG",
                quality=95,
            )

            # ------------------------------------------------
            # Convert annotations
            # ------------------------------------------------

            label_lines = []

            for item in annotations_by_image[
                image_id
            ]:

                ann = item[
                    "annotation"
                ]

                practical_class = item[
                    "practical_class"
                ]

                x, y, w, h = ann[
                    "bbox"
                ]

                yolo = bbox_to_yolo(
                    x,
                    y,
                    w,
                    h,
                    width,
                    height,
                )

                if yolo is None:

                    invalid_boxes += 1

                    continue

                class_id = class_to_id[
                    practical_class
                ]

                label_lines.append(
                    f"{class_id} "
                    f"{yolo[0]:.6f} "
                    f"{yolo[1]:.6f} "
                    f"{yolo[2]:.6f} "
                    f"{yolo[3]:.6f}"
                )

                class_counts[
                    practical_class
                ] += 1

                converted_boxes += 1

            # ------------------------------------------------
            # Remove images with zero valid boxes
            # ------------------------------------------------

            if not label_lines:

                output_image.unlink(
                    missing_ok=True
                )

                continue

            # ------------------------------------------------
            # Save YOLO label
            # ------------------------------------------------

            with open(
                output_label,
                "w",
                encoding="utf-8",
            ) as f:

                f.write(
                    "\n".join(
                        label_lines
                    )
                    + "\n"
                )

            processed_images += 1

    # ========================================================
    # WRITE data.yaml
    # ========================================================

    yaml_path = (
        OUTPUT_DIR
        / "data.yaml"
    )

    with open(
        yaml_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            f"path: "
            f"{OUTPUT_DIR.as_posix()}\n"
        )

        f.write(
            "train: images/train\n"
        )

        f.write(
            "val: images/val\n"
        )

        f.write(
            "test: images/test\n\n"
        )

        f.write(
            "names:\n"
        )

        for idx, name in enumerate(
            PRACTICAL_CLASSES
        ):

            f.write(
                f"  {idx}: {name}\n"
            )

    # ========================================================
    # WRITE CLASS MANIFEST
    # ========================================================

    manifest_path = (
        OUTPUT_DIR
        / "class_manifest.csv"
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "class_id,"
            "practical_class,"
            "annotation_count\n"
        )

        for idx, name in enumerate(
            PRACTICAL_CLASSES
        ):

            f.write(
                f"{idx},"
                f"{name},"
                f"{class_counts[name]}\n"
            )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 70)
    print(
        "TACO PRACTICAL 6-CLASS "
        "DATASET BUILD COMPLETE"
    )
    print("=" * 70)

    print(
        f"Source images:        "
        f"{len(images)}"
    )

    print(
        f"Selected images:      "
        f"{len(selected_image_ids)}"
    )

    print(
        f"Processed images:     "
        f"{processed_images}"
    )

    print(
        f"Selected annotations: "
        f"{selected_annotations}"
    )

    print(
        f"Converted boxes:      "
        f"{converted_boxes}"
    )

    print(
        f"Invalid boxes:        "
        f"{invalid_boxes}"
    )

    print(
        f"Missing images:       "
        f"{missing_images}"
    )

    print()
    print(
        "CLASS DISTRIBUTION"
    )

    print("-" * 70)

    for idx, name in enumerate(
        PRACTICAL_CLASSES
    ):

        print(
            f"{idx:2} "
            f"{name:<18} "
            f"{class_counts[name]}"
        )

    print()
    print(
        "EXIF ORIENTATIONS"
    )

    print("-" * 70)

    for orientation, count in sorted(
        orientation_counts.items()
    ):

        print(
            f"{orientation}: "
            f"{count}"
        )

    print()
    print("OUTPUT")
    print("-" * 70)

    print(
        OUTPUT_DIR
    )

    print(
        yaml_path
    )

    print(
        manifest_path
    )

    # ========================================================
    # FINAL PASS/FAIL
    # ========================================================

    expected_counts = {
        "plastic": 2334,
        "glass": 254,
        "metal": 459,
        "paper_cardboard": 488,
        "biological": 8,
        "other_trash": 1239,
    }

    counts_match = all(
        class_counts[name]
        == expected_counts[name]
        for name in PRACTICAL_CLASSES
    )

    if (
        processed_images == 1499
        and selected_annotations == 4782
        and converted_boxes == 4782
        and invalid_boxes == 0
        and missing_images == 0
        and counts_match
    ):

        print()
        print(
            "PASS - practical dataset "
            "build is clean."
        )

    else:

        print()
        print(
            "WARNING - final counts "
            "require inspection."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()