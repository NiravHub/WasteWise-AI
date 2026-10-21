from pathlib import Path
from collections import Counter
import json
import zipfile


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "data" / "external"
RAW = ROOT / "data" / "raw"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}


def print_header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def get_images(path):
    return [
        p
        for p in path.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
        and ".git" not in p.parts
    ]


# ============================================================
# 1. MAIN GARBAGE DATASET
# ============================================================

print_header("1. MAIN GARBAGE DATASET")

garbage = RAW / "garbage-dataset"

if garbage.exists():
    all_files = [
        p for p in garbage.rglob("*")
        if p.is_file()
    ]

    images = get_images(garbage)

    print(f"Path       : {garbage}")
    print(f"All files  : {len(all_files):,}")
    print(f"Images     : {len(images):,}")

    classes = Counter(p.parent.name for p in images)

    print("\nClasses:")
    for name, count in sorted(classes.items()):
        print(f"  {name:<20} {count:>7,}")
else:
    print(f"NOT FOUND: {garbage}")


# ============================================================
# 2. COCO
# ============================================================

print_header("2. COCO")

coco = EXTERNAL / "coco"

if coco.exists():

    print("Files:")

    for p in coco.iterdir():
        if p.is_file():
            size_gb = p.stat().st_size / (1024 ** 3)
            print(f"  {p.name:<30} {size_gb:.3f} GB")

    # -----------------------------
    # COCO JSON
    # -----------------------------

    json_file = coco / "instances_val2017.json"

    if json_file.exists():

        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            print("\nCOCO annotation information:")
            print(
                f"  Images      : "
                f"{len(data.get('images', [])):,}"
            )

            print(
                f"  Annotations : "
                f"{len(data.get('annotations', [])):,}"
            )

            print(
                f"  Categories  : "
                f"{len(data.get('categories', [])):,}"
            )

            print("\nCOCO categories:")

            for category in data.get("categories", []):
                print(
                    f"  {category['id']:>3}  "
                    f"{category['name']}"
                )

        except Exception as e:
            print(f"JSON ERROR: {e}")

    else:
        print("instances_val2017.json NOT FOUND")

    # -----------------------------
    # COCO ZIP
    # -----------------------------

    zip_file = coco / "val2017.zip"

    if zip_file.exists():

        try:
            with zipfile.ZipFile(zip_file, "r") as z:

                members = z.namelist()

                image_members = [
                    x
                    for x in members
                    if Path(x).suffix.lower()
                    in IMAGE_EXTENSIONS
                ]

                print("\nCOCO ZIP:")
                print(f"  ZIP members : {len(members):,}")
                print(f"  ZIP images  : {len(image_members):,}")

        except Exception as e:
            print(f"ZIP ERROR: {e}")

    else:
        print("val2017.zip NOT FOUND")

else:
    print(f"NOT FOUND: {coco}")


# ============================================================
# 3. TACO
# ============================================================

print_header("3. TACO")

taco = EXTERNAL / "taco"
taco_zip = taco / "TACO.zip"

if taco_zip.exists():

    size_gb = taco_zip.stat().st_size / (1024 ** 3)

    print(f"TACO ZIP size : {size_gb:.3f} GB")

    try:
        with zipfile.ZipFile(taco_zip, "r") as z:

            members = z.namelist()

            images = [
                x
                for x in members
                if Path(x).suffix.lower()
                in IMAGE_EXTENSIONS
            ]

            jsons = [
                x
                for x in members
                if Path(x).suffix.lower() == ".json"
            ]

            print(f"ZIP members : {len(members):,}")
            print(f"Images      : {len(images):,}")
            print(f"JSON files  : {len(jsons):,}")

            print("\nTACO JSON files:")

            for item in jsons[:50]:
                print(f"  {item}")

    except Exception as e:
        print(f"ZIP ERROR: {e}")

else:
    print("TACO.zip NOT FOUND")


# ============================================================
# 4. TRASHNET++
# ============================================================

print_header("4. TRASHNET++")

trash = EXTERNAL / "trashnet_plus_plus"

if trash.exists():

    all_files = [
        p
        for p in trash.rglob("*")
        if p.is_file()
        and ".git" not in p.parts
    ]

    images = get_images(trash)

    print(f"All files : {len(all_files):,}")
    print(f"Images    : {len(images):,}")

    print("\nDirectory structure containing images:")

    folders = Counter(
        p.parent.relative_to(trash).as_posix()
        for p in images
    )

    for folder, count in sorted(folders.items()):
        print(f"  {folder:<60} {count:>7,}")

else:
    print(f"NOT FOUND: {trash}")


# ============================================================
# 5. MEDICAL WASTE 4.0
# ============================================================

print_header("5. MEDICAL WASTE 4.0")

medical = EXTERNAL / "medical_waste_4_0"

if medical.exists():

    all_files = [
        p
        for p in medical.rglob("*")
        if p.is_file()
        and ".git" not in p.parts
    ]

    images = get_images(medical)

    print(f"All files : {len(all_files):,}")
    print(f"Images    : {len(images):,}")

    # Count leaf folders
    leaf_counts = Counter(
        p.parent.name
        for p in images
    )

    print("\nLeaf-folder image counts:")

    for name, count in sorted(leaf_counts.items()):
        print(f"  {name:<35} {count:>7,}")

else:
    print(f"NOT FOUND: {medical}")


# ============================================================
# COMPLETE
# ============================================================

print_header(
    "INVENTORY COMPLETE — NO FILES WERE MODIFIED"
)