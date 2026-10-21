from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import math

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "results"
    / "dataset_audit"
    / "visual_overlap_samples"
)

OUTPUT = (
    ROOT
    / "results"
    / "dataset_audit"
    / "taxonomy_contact_sheets"
)

OUTPUT.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}

THUMB_W = 220
THUMB_H = 180
COLS = 4


# ============================================================
# HELPERS
# ============================================================

def get_images(folder):
    if not folder.exists():
        return []

    return sorted(
        p
        for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def make_sheet(
    title,
    folders,
    output_name,
):
    """
    folders:
        list of (display_name, Path)
    """

    rows = []

    for display_name, folder in folders:
        images = get_images(folder)

        if not images:
            continue

        for image in images:
            rows.append(
                (
                    display_name,
                    image,
                )
            )

    if not rows:
        print(f"[WARNING] No images: {title}")
        return

    total = len(rows)
    grid_rows = math.ceil(total / COLS)

    title_height = 60
    cell_height = THUMB_H + 45

    sheet = Image.new(
        "RGB",
        (
            COLS * THUMB_W,
            title_height + grid_rows * cell_height,
        ),
        "white",
    )

    draw = ImageDraw.Draw(sheet)

    draw.text(
        (10, 15),
        title,
        fill="black",
    )

    for i, (label, image_path) in enumerate(rows):

        col = i % COLS
        row = i // COLS

        x = col * THUMB_W
        y = title_height + row * cell_height

        try:
            image = Image.open(image_path).convert("RGB")

            image.thumbnail(
                (
                    THUMB_W - 20,
                    THUMB_H - 20,
                )
            )

            canvas = Image.new(
                "RGB",
                (
                    THUMB_W,
                    THUMB_H,
                ),
                "white",
            )

            px = (
                THUMB_W - image.width
            ) // 2

            py = (
                THUMB_H - image.height
            ) // 2

            canvas.paste(
                image,
                (px, py),
            )

            sheet.paste(
                canvas,
                (x, y),
            )

            draw.text(
                (
                    x + 5,
                    y + THUMB_H + 5,
                ),
                label[:32],
                fill="black",
            )

        except Exception as e:
            print(
                f"[WARNING] Could not open "
                f"{image_path}: {e}"
            )

    output = OUTPUT / output_name

    sheet.save(
        output,
        quality=95,
    )

    print(f"[CREATED] {output}")


# ============================================================
# CLASS COMPARISONS
# ============================================================

garbage = INPUT / "garbage_dataset"
trashnet = INPUT / "trashnet_plus_plus"
medical = INPUT / "medical_waste_4_0"


# ------------------------------------------------------------
# PLASTIC
# ------------------------------------------------------------

make_sheet(
    "Plastic: Garbage Dataset vs TrashNet++",
    [
        (
            "Garbage plastic",
            garbage / "plastic",
        ),
        (
            "TrashNet Plastic",
            trashnet / "plastic",
        ),
    ],
    "01_plastic_comparison.jpg",
)


# ------------------------------------------------------------
# GLASS
# ------------------------------------------------------------

make_sheet(
    "Glass: Garbage Dataset vs TrashNet++",
    [
        (
            "Garbage glass",
            garbage / "glass",
        ),
        (
            "TrashNet Glass",
            trashnet / "glass",
        ),
    ],
    "02_glass_comparison.jpg",
)


# ------------------------------------------------------------
# METAL
# ------------------------------------------------------------

make_sheet(
    "Metal: Garbage Dataset vs TrashNet++",
    [
        (
            "Garbage metal",
            garbage / "metal",
        ),
        (
            "TrashNet Metal",
            trashnet / "metal",
        ),
    ],
    "03_metal_comparison.jpg",
)


# ------------------------------------------------------------
# BATTERY / E-WASTE
# ------------------------------------------------------------

make_sheet(
    "Battery and E-Waste",
    [
        (
            "Garbage battery",
            garbage / "battery",
        ),
        (
            "TrashNet E-wast",
            trashnet / "e-wast",
        ),
    ],
    "04_battery_ewaste_comparison.jpg",
)


# ------------------------------------------------------------
# MEDICAL
# ------------------------------------------------------------

make_sheet(
    "TrashNet Medical Waste",
    [
        (
            "TrashNet Medical Wast",
            trashnet / "medical_wast",
        ),
    ],
    "05_trashnet_medical.jpg",
)


# ------------------------------------------------------------
# GLOVES
# ------------------------------------------------------------

make_sheet(
    "Medical Gloves — Dataset A",
    [
        (
            "Latex pair",
            medical
            / "dataset_a"
            / "glove_pair_latex",
        ),
        (
            "Nitrile pair",
            medical
            / "dataset_a"
            / "glove_pair_nitrile",
        ),
        (
            "Surgery pair",
            medical
            / "dataset_a"
            / "glove_pair_surgery",
        ),
        (
            "Latex single",
            medical
            / "dataset_a"
            / "glove_single_latex",
        ),
        (
            "Nitrile single",
            medical
            / "dataset_a"
            / "glove_single_nitrile",
        ),
        (
            "Surgery single",
            medical
            / "dataset_a"
            / "glove_single_surgery",
        ),
    ],
    "06_medical_gloves_A.jpg",
)


# ------------------------------------------------------------
# MEDICAL OBJECTS
# ------------------------------------------------------------

make_sheet(
    "Medical Waste Objects — Dataset A",
    [
        (
            "Gauze",
            medical
            / "dataset_a"
            / "gauze",
        ),
        (
            "Medical cap",
            medical
            / "dataset_a"
            / "medical_cap",
        ),
        (
            "Medical glasses",
            medical
            / "dataset_a"
            / "medical_glasses",
        ),
        (
            "Shoe cover",
            medical
            / "dataset_a"
            / "shoe_cover_pair",
        ),
        (
            "Test tube",
            medical
            / "dataset_a"
            / "test_tube",
        ),
        (
            "Urine bag",
            medical
            / "dataset_a"
            / "urine_bag",
        ),
    ],
    "07_medical_objects_A.jpg",
)


# ------------------------------------------------------------
# GENERAL MATERIAL CLASSES
# ------------------------------------------------------------

make_sheet(
    "Garbage Dataset Core Classes",
    [
        ("Plastic", garbage / "plastic"),
        ("Metal", garbage / "metal"),
        ("Glass", garbage / "glass"),
        ("Battery", garbage / "battery"),
        ("Biological", garbage / "biological"),
        ("Cardboard", garbage / "cardboard"),
        ("Paper", garbage / "paper"),
        ("Trash", garbage / "trash"),
        ("Clothes", garbage / "clothes"),
        ("Shoes", garbage / "shoes"),
    ],
    "08_garbage_core_classes.jpg",
)


print("\n" + "=" * 70)
print("CONTACT SHEETS COMPLETE")
print("=" * 70)
print(f"Output: {OUTPUT}")
print("=" * 70)