import json
import csv
from pathlib import Path
from PIL import Image

ROOT = Path("data/external/taco/extracted/TACO/data")
OUT = Path("results/dataset_audit/taco")
OUT.mkdir(parents=True, exist_ok=True)

MASTER = ROOT / "annotations.json"

# ------------------------------------------------------------
# Load master annotations.json
# ------------------------------------------------------------
with open(MASTER, "r", encoding="utf-8") as f:
    master = json.load(f)

images = master["images"]

print(f"Master image records: {len(images)}")

rows = []
resolved = 0
missing = 0
dimension_mismatch = 0

for img in images:
    image_id = img["id"]
    file_name = img["file_name"]
    expected_w = img["width"]
    expected_h = img["height"]

    # EXACT path from master JSON
    image_path = ROOT / Path(file_name)

    status = "OK"

    if not image_path.exists():
        status = "MISSING"
        missing += 1
    else:
        try:
            with Image.open(image_path) as im:
                actual_w, actual_h = im.size

            if actual_w != expected_w or actual_h != expected_h:
                status = "DIMENSION_MISMATCH"
                dimension_mismatch += 1
            else:
                resolved += 1

        except Exception as e:
            status = f"IMAGE_ERROR:{e}"

    rows.append({
        "image_id": image_id,
        "file_name": file_name,
        "width": expected_w,
        "height": expected_h,
        "image_path": str(image_path.resolve()),
        "status": status
    })

# ------------------------------------------------------------
# Save manifest
# ------------------------------------------------------------
manifest_path = OUT / "taco_image_manifest_corrected.csv"

with open(manifest_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "image_id",
            "file_name",
            "width",
            "height",
            "image_path",
            "status"
        ]
    )
    writer.writeheader()
    writer.writerows(rows)

print()
print("=" * 70)
print("CORRECTED TACO MANIFEST")
print("=" * 70)
print(f"Total images:          {len(images)}")
print(f"Resolved correctly:    {resolved}")
print(f"Missing:               {missing}")
print(f"Dimension mismatches:  {dimension_mismatch}")
print()
print(f"Manifest:")
print(manifest_path)
