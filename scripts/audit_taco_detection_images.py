from pathlib import Path
import csv
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "results" / "dataset_audit" / "taco"

ANNOTATIONS = AUDIT / "taco_clean_annotations.csv"
CATEGORIES = AUDIT / "taco_categories.csv"

SELECTED_THRESHOLD = 50

# ------------------------------------------------------------
# Load categories
# ------------------------------------------------------------

categories = {}

with open(CATEGORIES, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        categories[int(row["category_id"])] = row["name"]

# ------------------------------------------------------------
# Load annotations
# ------------------------------------------------------------

class_annotations = Counter()
class_images = defaultdict(set)
image_classes = defaultdict(set)
image_annotation_count = Counter()

with open(ANNOTATIONS, "r", encoding="utf-8") as f:

    for row in csv.DictReader(f):

        image_id = row["image_id"]
        category_id = int(row["category_id"])

        class_annotations[category_id] += 1
        class_images[category_id].add(image_id)

        image_classes[image_id].add(category_id)
        image_annotation_count[image_id] += 1

# ------------------------------------------------------------
# Select classes with >= 50 annotations
# ------------------------------------------------------------

selected_classes = {
    category_id
    for category_id, count in class_annotations.items()
    if count >= SELECTED_THRESHOLD
}

selected_images = {
    image_id
    for image_id, classes in image_classes.items()
    if classes & selected_classes
}

# ------------------------------------------------------------
# Statistics
# ------------------------------------------------------------

print("=" * 75)
print("TACO DETECTION IMAGE DISTRIBUTION AUDIT")
print("=" * 75)

print(f"\nTotal images with annotations: {len(image_classes):,}")
print(f"Total annotations:             {sum(class_annotations.values()):,}")
print(f"Selected classes:              {len(selected_classes)}")
print(f"Selected images:               {len(selected_images):,}")

print("\n" + "=" * 75)
print("CLASS DISTRIBUTION")
print("=" * 75)

rows = []

for category_id in sorted(
    selected_classes,
    key=lambda x: class_annotations[x],
    reverse=True
):

    rows.append({
        "category_id": category_id,
        "name": categories.get(
            category_id,
            f"UNKNOWN_{category_id}"
        ),
        "annotations": class_annotations[category_id],
        "images": len(class_images[category_id]),
    })

for row in rows:

    print(
        f"{row['category_id']:>2} | "
        f"{row['annotations']:>4} annotations | "
        f"{row['images']:>4} images | "
        f"{row['name']}"
    )

# ------------------------------------------------------------
# Image-level annotation density
# ------------------------------------------------------------

selected_annotation_counts = []

for image_id in selected_images:

    count = sum(
        1
        for category_id in image_classes[image_id]
        if category_id in selected_classes
    )

    selected_annotation_counts.append(
        image_annotation_count[image_id]
    )

print("\n" + "=" * 75)
print("IMAGE-LEVEL STATISTICS")
print("=" * 75)

if selected_annotation_counts:

    print(
        f"Selected images:              "
        f"{len(selected_annotation_counts):,}"
    )

    print(
        f"Annotations/image (mean):     "
        f"{sum(selected_annotation_counts) / len(selected_annotation_counts):.2f}"
    )

    print(
        f"Annotations/image (max):      "
        f"{max(selected_annotation_counts)}"
    )

# ------------------------------------------------------------
# Multi-class images
# ------------------------------------------------------------

multi_class = 0
single_class = 0

for image_id in selected_images:

    selected_in_image = (
        image_classes[image_id]
        & selected_classes
    )

    if len(selected_in_image) > 1:
        multi_class += 1
    else:
        single_class += 1

print(
    f"\nSingle selected class images:  {single_class:,}"
)

print(
    f"Multi selected class images:   {multi_class:,}"
)

# ------------------------------------------------------------
# Class co-occurrence
# ------------------------------------------------------------

cooccurrence = Counter()

for image_id in selected_images:

    classes = sorted(
        image_classes[image_id]
        & selected_classes
    )

    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):

            pair = (
                classes[i],
                classes[j]
            )

            cooccurrence[pair] += 1

print("\n" + "=" * 75)
print("TOP CLASS CO-OCCURRENCES")
print("=" * 75)

for (a, b), count in cooccurrence.most_common(20):

    print(
        f"{categories[a]} + "
        f"{categories[b]}: "
        f"{count} images"
    )

# ------------------------------------------------------------
# Final recommendation
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("RECOMMENDATION")
print("=" * 75)

print(
    "\nUse image-level splitting."
)

print(
    "Do NOT split individual bounding boxes independently."
)

print(
    "Keep the original TACO source untouched."
)

print(
    "\nNext step: generate YOLO-format train/val/test "
    "sets from the selected classes."
)
