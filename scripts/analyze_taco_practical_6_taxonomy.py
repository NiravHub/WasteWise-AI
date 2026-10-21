from pathlib import Path
import json
from collections import Counter, defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ANNOTATIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "taco"
    / "extracted"
    / "TACO"
    / "data"
    / "annotations.json"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "dataset_audit" / "taco"
OUTPUT_FILE = OUTPUT_DIR / "practical_6_taxonomy_analysis.txt"

# ---------------------------------------------------------
# TACO -> Practical 6-class taxonomy
# ---------------------------------------------------------

SOURCE_TO_PRACTICAL = {
"Crisp packet": "plastic",
"Paper cup": "paper_cardboard",
"Rope & strings": "other_trash",
    # Plastic
    "Plastic film": "plastic",
    "Other plastic wrapper": "plastic",
    "Clear plastic bottle": "plastic",
    "Other plastic bottle": "plastic",
    "Other plastic": "plastic",
    "Plastic bottle cap": "plastic",
    "Metal bottle cap": "plastic",
    "Plastic straw": "plastic",
    "Paper straw": "plastic",
    "Disposable plastic cup": "plastic",
    "Foam cup": "plastic",
    "Other plastic cup": "plastic",
    "Plastic lid": "plastic",
    "Metal lid": "plastic",
    "Garbage bag": "plastic",
    "Single-use carrier bag": "plastic",
    "Polypropylene bag": "plastic",
    "Disposable food container": "plastic",
    "Foam food container": "plastic",
    "Other plastic container": "plastic",
    "Tupperware": "plastic",
    "Styrofoam piece": "plastic",
    "Plastic utensils": "plastic",
    "Spread tub": "plastic",
    "Plastic glooves": "plastic",

    # Glass
    "Glass bottle": "glass",
    "Broken glass": "glass",
    "Glass jar": "glass",
    "Glass cup": "glass",

    # Metal
    "Drink can": "metal",
    "Food Can": "metal",
    "Aerosol": "metal",
    "Scrap metal": "metal",
    "Pop tab": "metal",
    "Aluminium foil": "metal",
    "Six pack rings": "metal",

    # Paper / cardboard
    "Normal paper": "paper_cardboard",
    "Magazine paper": "paper_cardboard",
    "Wrapping paper": "paper_cardboard",
    "Tissues": "paper_cardboard",
    "Other carton": "paper_cardboard",
    "Drink carton": "paper_cardboard",
    "Corrugated carton": "paper_cardboard",
    "Meal carton": "paper_cardboard",
    "Egg carton": "paper_cardboard",
    "Pizza box": "paper_cardboard",
    "Paper bag": "paper_cardboard",

    # Biological
    "Food waste": "biological",

    # Other / residual trash
    "Cigarette": "other_trash",
    "Unlabeled litter": "other_trash",
    "Aluminium blister pack": "other_trash",
    "Carded blister pack": "other_trash",
    "Shoe": "other_trash",
    "Squeezable tube": "other_trash",
    "Toilet tube": "other_trash",
}

PRACTICAL_CLASSES = [
    "plastic",
    "glass",
    "metal",
    "paper_cardboard",
    "biological",
    "other_trash",
]


def main():
    if not ANNOTATIONS_FILE.exists():
        raise FileNotFoundError(
            f"TACO annotations not found:\n{ANNOTATIONS_FILE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    categories = {
        c["id"]: c["name"]
        for c in data["categories"]
    }

    images = data["images"]
    annotations = data["annotations"]

    print("=" * 70)
    print("TACO PRACTICAL 6-CLASS TAXONOMY ANALYSIS")
    print("=" * 70)

    print(f"Source images:       {len(images)}")
    print(f"Source annotations:  {len(annotations)}")
    print(f"Source categories:   {len(categories)}")

    # -----------------------------------------------------
    # Category-level statistics
    # -----------------------------------------------------

    source_annotation_counts = Counter()
    source_image_ids = defaultdict(set)

    for ann in annotations:
        category_name = categories[ann["category_id"]]
        source_annotation_counts[category_name] += 1
        source_image_ids[category_name].add(ann["image_id"])

    annotated_source_classes = set(source_annotation_counts)

    unmapped = sorted(
        annotated_source_classes - set(SOURCE_TO_PRACTICAL)
    )

    mapped_target_counts = Counter()
    mapped_target_images = defaultdict(set)

    for ann in annotations:
        source_name = categories[ann["category_id"]]

        if source_name not in SOURCE_TO_PRACTICAL:
            continue

        target = SOURCE_TO_PRACTICAL[source_name]

        mapped_target_counts[target] += 1
        mapped_target_images[target].add(ann["image_id"])

    # -----------------------------------------------------
    # Image-level practical classes
    # -----------------------------------------------------

    image_target_classes = defaultdict(set)

    for ann in annotations:
        source_name = categories[ann["category_id"]]

        if source_name in SOURCE_TO_PRACTICAL:
            target = SOURCE_TO_PRACTICAL[source_name]
            image_target_classes[ann["image_id"]].add(target)

    multi_class_images = sum(
        1
        for classes in image_target_classes.values()
        if len(classes) > 1
    )

    single_class_images = sum(
        1
        for classes in image_target_classes.values()
        if len(classes) == 1
    )

    # -----------------------------------------------------
    # Mapping statistics
    # -----------------------------------------------------

    total_annotations = len(annotations)
    mapped_annotations = sum(mapped_target_counts.values())

    retention = (
        mapped_annotations / total_annotations * 100
        if total_annotations
        else 0
    )

    max_count = max(mapped_target_counts.values())
    min_count = min(mapped_target_counts.values())

    imbalance_ratio = (
        max_count / min_count
        if min_count
        else float("inf")
    )

    # -----------------------------------------------------
    # Generate report
    # -----------------------------------------------------

    lines = []

    lines.append("=" * 70)
    lines.append("TACO PRACTICAL 6-CLASS TAXONOMY ANALYSIS")
    lines.append("=" * 70)
    lines.append("")

    lines.append("SOURCE DATA")
    lines.append("-" * 70)
    lines.append(f"Images:                 {len(images)}")
    lines.append(f"Annotations:            {len(annotations)}")
    lines.append(f"Category definitions:   {len(categories)}")
    lines.append(f"Annotated categories:   {len(annotated_source_classes)}")
    lines.append("")

    lines.append("PRACTICAL TAXONOMY")
    lines.append("-" * 70)

    for idx, cls in enumerate(PRACTICAL_CLASSES):
        count = mapped_target_counts[cls]
        image_count = len(mapped_target_images[cls])

        percentage = (
            count / total_annotations * 100
            if total_annotations
            else 0
        )

        lines.append(
            f"{idx}: {cls:<18} "
            f"annotations={count:<5} "
            f"images={image_count:<4} "
            f"share={percentage:6.2f}%"
        )

    lines.append("")

    lines.append("DATASET RETENTION")
    lines.append("-" * 70)
    lines.append(f"Original annotations:   {total_annotations}")
    lines.append(f"Mapped annotations:     {mapped_annotations}")
    lines.append(f"Unmapped annotations:   {total_annotations - mapped_annotations}")
    lines.append(f"Retention:              {retention:.2f}%")
    lines.append("")

    lines.append("CLASS IMBALANCE")
    lines.append("-" * 70)
    lines.append(f"Largest class:          {max_count}")
    lines.append(f"Smallest class:         {min_count}")
    lines.append(f"Imbalance ratio:        {imbalance_ratio:.2f}x")
    lines.append("")

    lines.append("IMAGE DISTRIBUTION")
    lines.append("-" * 70)
    lines.append(f"Images with mapped labels: {len(image_target_classes)}")
    lines.append(f"Single-class images:       {single_class_images}")
    lines.append(f"Multi-class images:        {multi_class_images}")
    lines.append("")

    lines.append("SOURCE CATEGORY MAPPING")
    lines.append("-" * 70)

    for practical_class in PRACTICAL_CLASSES:
        lines.append("")
        lines.append(f"[{practical_class}]")

        mapped_sources = sorted(
            name
            for name, target in SOURCE_TO_PRACTICAL.items()
            if target == practical_class
        )

        for source_name in mapped_sources:
            count = source_annotation_counts.get(source_name, 0)
            lines.append(
                f"  {source_name:<30} {count}"
            )

    lines.append("")
    lines.append("UNMAPPED ANNOTATED CATEGORIES")
    lines.append("-" * 70)

    if unmapped:
        for name in unmapped:
            lines.append(
                f"  {name}: {source_annotation_counts[name]}"
            )
    else:
        lines.append("  NONE")

    lines.append("")
    lines.append("ZERO-INSTANCE CATEGORY DEFINITIONS")
    lines.append("-" * 70)

    zero_instance = sorted(
        set(categories.values()) - annotated_source_classes
    )

    if zero_instance:
        for name in zero_instance:
            lines.append(f"  {name}")
    else:
        lines.append("  NONE")

    lines.append("")
    lines.append("FINAL AUDIT")
    lines.append("-" * 70)

    if unmapped:
        lines.append(
            "REVIEW REQUIRED - annotated TACO categories are unmapped."
        )
    else:
        lines.append(
            "PASS - every annotated TACO category is mapped "
            "to one practical class."
        )

    if mapped_annotations == total_annotations:
        lines.append(
            "PASS - 100% annotation retention."
        )
    else:
        lines.append(
            f"WARNING - {total_annotations - mapped_annotations} "
            "annotations would be excluded."
        )

    lines.append("")
    lines.append("=" * 70)

    report = "\n".join(lines)

    print()
    print(report)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(report)

    print()
    print(f"Report saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()