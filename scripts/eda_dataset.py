from pathlib import Path
from collections import Counter
import random

import matplotlib.pyplot as plt
from PIL import Image

# ============================================================
# WasteWise AI - Exploratory Data Analysis
# ============================================================

DATASET_DIR = Path("data/raw/garbage-dataset")
FIGURES_DIR = Path("results/figures")

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

RANDOM_SEED = 42

random.seed(RANDOM_SEED)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Validate dataset
# ------------------------------------------------------------

if not DATASET_DIR.exists():
    print(f"ERROR: Dataset not found at {DATASET_DIR.resolve()}")
    raise SystemExit(1)

# ------------------------------------------------------------
# Discover classes
# ------------------------------------------------------------

class_dirs = sorted(
    [directory for directory in DATASET_DIR.iterdir() if directory.is_dir()]
)

classes = [directory.name for directory in class_dirs]

print("=" * 70)
print("WASTEWISE AI - EXPLORATORY DATA ANALYSIS")
print("=" * 70)

print("\nClasses:")
for class_name in classes:
    print(f"  - {class_name}")

# ------------------------------------------------------------
# Collect image paths
# ------------------------------------------------------------

class_images = {}

for class_dir in class_dirs:

    images = [
        path
        for path in class_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    class_images[class_dir.name] = images

# ------------------------------------------------------------
# 1. CLASS DISTRIBUTION
# ------------------------------------------------------------

class_counts = {
    class_name: len(class_images[class_name])
    for class_name in classes
}

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

for class_name in classes:
    print(f"{class_name:<20} {class_counts[class_name]:>6}")

# Sort classes by number of images
sorted_classes = sorted(
    class_counts,
    key=class_counts.get,
    reverse=True,
)

sorted_counts = [
    class_counts[class_name]
    for class_name in sorted_classes
]

plt.figure(figsize=(12, 7))

bars = plt.bar(
    sorted_classes,
    sorted_counts,
)

plt.title(
    "WasteWise AI Dataset - Class Distribution",
    fontsize=16,
    fontweight="bold",
)

plt.xlabel("Waste Class")
plt.ylabel("Number of Images")

plt.xticks(
    rotation=35,
    ha="right",
)

# Add values above bars
for bar, count in zip(bars, sorted_counts):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        str(count),
        ha="center",
        va="bottom",
        fontsize=9,
    )

plt.tight_layout()

distribution_path = FIGURES_DIR / "class_distribution.png"

plt.savefig(
    distribution_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(f"\nSaved: {distribution_path}")

# ------------------------------------------------------------
# 2. REPRESENTATIVE IMAGE GRID
# ------------------------------------------------------------

print("\nCreating representative image grid...")

fig, axes = plt.subplots(
    2,
    5,
    figsize=(16, 7),
)

axes = axes.flatten()

for index, class_name in enumerate(classes):

    image_paths = class_images[class_name]

    if not image_paths:
        axes[index].axis("off")
        continue

    image_path = random.choice(image_paths)

    try:
        image = Image.open(image_path).convert("RGB")

        axes[index].imshow(image)
        axes[index].set_title(
            f"{class_name}\n({class_counts[class_name]} images)",
            fontsize=11,
            fontweight="bold",
        )

    except Exception as error:

        axes[index].text(
            0.5,
            0.5,
            "Image Error",
            ha="center",
            va="center",
        )

        print(
            f"Warning: Could not load {image_path}: {error}"
        )

    axes[index].axis("off")

fig.suptitle(
    "WasteWise AI - Representative Samples",
    fontsize=18,
    fontweight="bold",
)

plt.tight_layout()

sample_path = FIGURES_DIR / "representative_samples.png"

plt.savefig(
    sample_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(f"Saved: {sample_path}")

# ------------------------------------------------------------
# 3. IMAGE DIMENSION DISTRIBUTION
# ------------------------------------------------------------

print("\nAnalyzing image dimensions...")

dimension_counts = Counter()

for class_name in classes:

    for image_path in class_images[class_name]:

        try:

            with Image.open(image_path) as image:

                width, height = image.size

                dimension_counts[
                    (width, height)
                ] += 1

        except Exception:
            continue

print(
    f"Unique image dimensions: "
    f"{len(dimension_counts)}"
)

# Most common dimensions
top_dimensions = dimension_counts.most_common(15)

print("\nMost common dimensions:")

for (width, height), count in top_dimensions:

    print(
        f"{width} x {height:<6} : {count}"
    )

# ------------------------------------------------------------
# 4. ASPECT RATIO DISTRIBUTION
# ------------------------------------------------------------

aspect_ratios = []

for class_name in classes:

    for image_path in class_images[class_name]:

        try:

            with Image.open(image_path) as image:

                width, height = image.size

                if height > 0:

                    aspect_ratios.append(
                        width / height
                    )

        except Exception:
            continue

plt.figure(figsize=(10, 6))

plt.hist(
    aspect_ratios,
    bins=50,
)

plt.title(
    "WasteWise AI - Image Aspect Ratio Distribution",
    fontsize=16,
    fontweight="bold",
)

plt.xlabel("Width / Height")
plt.ylabel("Number of Images")

plt.tight_layout()

aspect_path = (
    FIGURES_DIR /
    "aspect_ratio_distribution.png"
)

plt.savefig(
    aspect_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(f"\nSaved: {aspect_path}")

# ------------------------------------------------------------
# FINAL
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("EDA COMPLETE")
print("=" * 70)

print("\nGenerated figures:")

print(
    f"  {distribution_path}"
)

print(
    f"  {sample_path}"
)

print(
    f"  {aspect_path}"
)

print("\nDataset was not modified.")