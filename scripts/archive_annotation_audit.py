from pathlib import Path
from collections import Counter
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "data" / "external"
OUTPUT = ROOT / "results" / "dataset_audit"
OUTPUT.mkdir(parents=True, exist_ok=True)


def audit_coco():
    print("\n" + "=" * 80)
    print("COCO ARCHIVE + ANNOTATION AUDIT")
    print("=" * 80)

    folder = EXTERNAL / "coco"
    json_path = folder / "instances_val2017.json"
    zip_path = folder / "val2017.zip"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    print(f"\nCOCO images in JSON       : {len(images):,}")
    print(f"COCO annotations           : {len(annotations):,}")
    print(f"COCO categories            : {len(categories):,}")

    category_map = {
        c["id"]: c["name"]
        for c in categories
    }

    annotation_counts = Counter(
        ann["category_id"]
        for ann in annotations
    )

    print("\nTop COCO categories by annotation count:")

    for category_id, count in annotation_counts.most_common(20):
        print(
            f"  {category_map.get(category_id, 'UNKNOWN'):<25}"
            f"{count:>8,}"
        )

    # Images referenced by annotations
    annotated_image_ids = {
        ann["image_id"]
        for ann in annotations
    }

    image_ids = {
        img["id"]
        for img in images
    }

    missing_annotation_images = image_ids - annotated_image_ids

    print(
        f"\nImages with >=1 annotation : "
        f"{len(annotated_image_ids):,}"
    )

    print(
        f"Images without annotations : "
        f"{len(missing_annotation_images):,}"
    )

    # Check ZIP
    with zipfile.ZipFile(zip_path, "r") as z:
        members = z.namelist()

    zip_files = [
        x for x in members
        if not x.endswith("/")
    ]

    zip_images = [
        x for x in zip_files
        if Path(x).suffix.lower()
        in {".jpg", ".jpeg", ".png", ".webp"}
    ]

    print(f"\nZIP total members           : {len(zip_files):,}")
    print(f"ZIP image members           : {len(zip_images):,}")

    json_filenames = {
        Path(img["file_name"]).name
        for img in images
    }

    zip_filenames = {
        Path(name).name
        for name in zip_images
    }

    json_not_in_zip = json_filenames - zip_filenames
    zip_not_in_json = zip_filenames - json_filenames

    print(
        f"\nJSON images missing from ZIP: "
        f"{len(json_not_in_zip):,}"
    )

    print(
        f"ZIP images missing from JSON: "
        f"{len(zip_not_in_json):,}"
    )

    # Useful broad OOD groups
    ood_keywords = {
        "people": {
            "person"
        },
        "animals": {
            "bird", "cat", "dog", "horse", "sheep",
            "cow", "elephant", "bear", "zebra", "giraffe",
            "teddy bear"
        },
        "vehicles": {
            "bicycle", "car", "motorcycle", "airplane",
            "bus", "train", "truck", "boat"
        },
        "electronics": {
            "laptop", "mouse", "remote", "keyboard",
            "cell phone", "tv", "microwave", "oven",
            "toaster", "refrigerator"
        },
        "food": {
            "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza",
            "donut", "cake"
        },
    }

    ood_summary = {}

    for group, names in ood_keywords.items():

        ids = {
            cid
            for cid, name in category_map.items()
            if name in names
        }

        count = sum(
            value
            for cid, value in annotation_counts.items()
            if cid in ids
        )

        ood_summary[group] = {
            "categories": sorted(names),
            "annotation_count": count,
        }

    print("\nPotential OOD groups:")

    for group, info in ood_summary.items():
        print(
            f"  {group:<15}"
            f"{info['annotation_count']:>8,} annotations"
        )

    report = {
        "dataset": "COCO val2017",
        "images": len(images),
        "annotations": len(annotations),
        "categories": len(categories),
        "annotated_images": len(annotated_image_ids),
        "unannotated_images": len(missing_annotation_images),
        "zip_members": len(zip_files),
        "zip_images": len(zip_images),
        "json_not_in_zip": sorted(json_not_in_zip),
        "zip_not_in_json": sorted(zip_not_in_json),
        "category_annotation_counts": {
            category_map.get(k, str(k)): v
            for k, v in annotation_counts.items()
        },
        "ood_summary": ood_summary,
    }

    output = OUTPUT / "coco_annotation_audit.json"

    with open(output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved: {output}")


def audit_taco():
    print("\n" + "=" * 80)
    print("TACO ARCHIVE + ANNOTATION AUDIT")
    print("=" * 80)

    zip_path = EXTERNAL / "taco" / "TACO.zip"

    with zipfile.ZipFile(zip_path, "r") as z:

        members = z.namelist()

        json_members = [
            x for x in members
            if x.lower().endswith(".json")
        ]

        image_members = [
            x for x in members
            if Path(x).suffix.lower()
            in {".jpg", ".jpeg", ".png", ".webp"}
        ]

        print(f"\nZIP total members : {len(members):,}")
        print(f"JSON files        : {len(json_members):,}")
        print(f"Image files       : {len(image_members):,}")

        all_categories = Counter()
        all_images = set()
        all_annotations = 0

        json_reports = []

        for json_member in json_members:

            try:
                raw = z.read(json_member)
                data = json.loads(raw.decode("utf-8"))

                images = data.get("images", [])
                annotations = data.get("annotations", [])
                categories = data.get("categories", [])

                all_annotations += len(annotations)

                for image in images:
                    file_name = image.get("file_name")

                    if file_name:
                        all_images.add(file_name)

                for category in categories:
                    name = category.get("name", "UNKNOWN")
                    all_categories[name] += 1

                json_reports.append({
                    "file": json_member,
                    "images": len(images),
                    "annotations": len(annotations),
                    "categories": len(categories),
                })

            except Exception as e:

                json_reports.append({
                    "file": json_member,
                    "error": str(e),
                })

    print(f"\nImages referenced by JSON : {len(all_images):,}")
    print(f"Total annotations         : {all_annotations:,}")

    print("\nTACO category definitions:")

    for category, count in sorted(
        all_categories.items(),
        key=lambda x: x[0].lower()
    ):
        print(f"  {category:<50} {count}")

    report = {
        "dataset": "TACO",
        "zip_members": len(members),
        "json_files": len(json_members),
        "image_files_in_zip": len(image_members),
        "images_referenced_by_json": len(all_images),
        "total_annotations": all_annotations,
        "category_definitions": dict(all_categories),
        "json_reports": json_reports,
    }

    output = OUTPUT / "taco_annotation_audit.json"

    with open(output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved: {output}")


def main():
    print("=" * 80)
    print("WASTEWISE AI — ARCHIVE & ANNOTATION AUDIT")
    print("=" * 80)

    audit_coco()
    audit_taco()

    print("\n" + "=" * 80)
    print("ARCHIVE AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()