from pathlib import Path
import json
import zipfile
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]

COCO_DIR = ROOT / "data" / "external" / "coco"
COCO_IMAGES = COCO_DIR / "val2017" / "val2017"
COCO_JSON = COCO_DIR / "instances_val2017.json"
COCO_ZIP = COCO_DIR / "val2017.zip"

TACO_DIR = ROOT / "data" / "external" / "taco" / "extracted" / "TACO"
TACO_ZIP = ROOT / "data" / "external" / "taco" / "TACO.zip"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"
}

print("=" * 70)
print("POST-EXTRACTION DATASET INTEGRITY AUDIT")
print("=" * 70)


# ================================================================
# COCO
# ================================================================

print("\n" + "=" * 70)
print("COCO VAL2017")
print("=" * 70)

coco_images = [
    p for p in COCO_IMAGES.rglob("*")
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
]

print(f"Extracted image files : {len(coco_images):,}")

with open(COCO_JSON, "r", encoding="utf-8") as f:
    coco = json.load(f)

coco_json_images = coco.get("images", [])
coco_annotations = coco.get("annotations", [])
coco_categories = coco.get("categories", [])

print(f"JSON image records     : {len(coco_json_images):,}")
print(f"Annotations            : {len(coco_annotations):,}")
print(f"Categories             : {len(coco_categories):,}")

json_filenames = {
    item["file_name"]
    for item in coco_json_images
    if "file_name" in item
}

disk_filenames = {
    p.name
    for p in coco_images
}

missing_images = sorted(json_filenames - disk_filenames)
orphan_images = sorted(disk_filenames - json_filenames)

print(f"Missing JSON images    : {len(missing_images):,}")
print(f"Orphan disk images     : {len(orphan_images):,}")

if not missing_images and not orphan_images:
    print("COCO alignment         : PASS")
else:
    print("COCO alignment         : REVIEW")


# ================================================================
# COCO ZIP
# ================================================================

print("\nChecking COCO ZIP...")

with zipfile.ZipFile(COCO_ZIP, "r") as z:
    members = z.namelist()

zip_image_members = [
    x for x in members
    if Path(x).suffix.lower() in IMAGE_EXTENSIONS
]

print(f"ZIP image members      : {len(zip_image_members):,}")


# ================================================================
# TACO
# ================================================================

print("\n" + "=" * 70)
print("TACO")
print("=" * 70)

taco_images = [
    p for p in TACO_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
]

print(f"Extracted image files  : {len(taco_images):,}")

taco_jsons = list(TACO_DIR.rglob("annotations.json"))

print(f"Annotation JSON files  : {len(taco_jsons):,}")

total_taco_json_images = 0
total_taco_annotations = 0

taco_referenced_paths = []
taco_categories = Counter()

for json_path in sorted(taco_jsons):

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[ERROR] Could not read {json_path}: {e}")
        continue

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    total_taco_json_images += len(images)
    total_taco_annotations += len(annotations)

    for image in images:
        if "file_name" in image:
            taco_referenced_paths.append(
                (json_path.parent, image["file_name"])
            )

    for category in categories:
        if "name" in category:
            taco_categories[category["name"]] += 1

print(f"JSON image records     : {total_taco_json_images:,}")
print(f"Annotations            : {total_taco_annotations:,}")
print(f"Unique category names  : {len(taco_categories):,}")


# ================================================================
# TACO ZIP
# ================================================================

print("\nChecking TACO ZIP...")

with zipfile.ZipFile(TACO_ZIP, "r") as z:
    members = z.namelist()

zip_images = [
    x for x in members
    if Path(x).suffix.lower() in IMAGE_EXTENSIONS
]

zip_jsons = [
    x for x in members
    if x.lower().endswith(".json")
]

print(f"ZIP image members      : {len(zip_images):,}")
print(f"ZIP JSON files         : {len(zip_jsons):,}")


# ================================================================
# TACO image reference resolution
# ================================================================

print("\nResolving TACO annotation image references...")

resolved = 0
unresolved = 0

for parent, file_name in taco_referenced_paths:

    candidate = parent / file_name

    if candidate.exists():
        resolved += 1
    else:
        unresolved += 1

print(f"Resolved references    : {resolved:,}")
print(f"Unresolved references  : {unresolved:,}")


# ================================================================
# Summary
# ================================================================

print("\n" + "=" * 70)
print("AUDIT SUMMARY")
print("=" * 70)

print("\nCOCO")
print(f"  Extracted images     : {len(coco_images):,}")
print(f"  JSON image records   : {len(coco_json_images):,}")
print(f"  Annotations          : {len(coco_annotations):,}")
print(f"  Categories           : {len(coco_categories):,}")
print(f"  Missing images       : {len(missing_images):,}")
print(f"  Orphan images        : {len(orphan_images):,}")

print("\nTACO")
print(f"  Extracted images     : {len(taco_images):,}")
print(f"  JSON image records   : {total_taco_json_images:,}")
print(f"  Annotations          : {total_taco_annotations:,}")
print(f"  Annotation JSONs     : {len(taco_jsons):,}")
print(f"  ZIP image members    : {len(zip_images):,}")
print(f"  Resolved references : {resolved:,}")
print(f"  Unresolved refs     : {unresolved:,}")

print("\n" + "=" * 70)
print("NO DATASET FILES WERE MODIFIED")
print("=" * 70)