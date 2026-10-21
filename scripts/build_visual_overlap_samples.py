from pathlib import Path
import random
import shutil
import csv

ROOT = Path(__file__).resolve().parents[1]

OUTPUT = (
    ROOT
    / "results"
    / "dataset_audit"
    / "visual_overlap_samples"
)

OUTPUT.mkdir(parents=True, exist_ok=True)

SEED = 42
SAMPLES_PER_CLASS = 12

random.seed(SEED)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}


# ============================================================
# DATASET ROOTS
# ============================================================

GARBAGE = ROOT / "data" / "raw" / "garbage-dataset"

TRASHNET = (
    ROOT
    / "data"
    / "external"
    / "trashnet_plus_plus"
    / "TrashNet"
)

MEDICAL = (
    ROOT
    / "data"
    / "external"
    / "medical_waste_4_0"
    / "Medical-Waste-4.0-Dataset"
    / "Medical Waste 4.0"
)


# ============================================================
# CLASSES
# ============================================================

GARBAGE_CLASSES = [
    "plastic",
    "metal",
    "glass",
    "battery",
    "biological",
    "cardboard",
    "paper",
    "trash",
    "clothes",
    "shoes",
]

TRASHNET_CLASSES = [
    "Plastic",
    "Metal",
    "Glass",
    "Can",
    "E-wast",
    "Medical Wast",
    "Cable",
]

MEDICAL_CLASSES = [
    "gauze",
    "gauzes",
    "glove_pair_latex",
    "glove_pair_nitrile",
    "glove_pair_surgery",
    "glove_single_latex",
    "glove_single_nitrile",
    "glove_single_surgery",
    "medical_cap",
    "medical_filter",
    "medical_glasses",
    "shoe_cover_pair",
    "shoe_cover_single",
    "test_tube",
    "urine_bag",
]


# ============================================================
# HELPERS
# ============================================================

def get_images(folder):
    if not folder.exists():
        return []

    return [
        p
        for p in folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def sample_images(images, n):
    if len(images) <= n:
        return sorted(images)

    return sorted(random.sample(images, n))


def safe_name(value):
    return (
        value
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("+", "plus")
    )


# ============================================================
# MANIFEST
# ============================================================

rows = []


def copy_samples(
    source,
    label,
    images,
    destination,
    extra_metadata=None,
):
    if not images:
        print(
            f"[WARNING] No images found: "
            f"{source} / {label}"
        )
        return

    selected = sample_images(
        images,
        SAMPLES_PER_CLASS,
    )

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"{source:20s} | "
        f"{label:25s} | "
        f"{len(images):5d} images | "
        f"sample {len(selected)}"
    )

    for index, image in enumerate(
        selected,
        start=1,
    ):

        destination_file = (
            destination
            / f"{index:02d}_{image.name}"
        )

        shutil.copy2(
            image,
            destination_file,
        )

        row = {
            "source": source,
            "label": label,
            "total_images": len(images),
            "sample_index": index,
            "original_path": str(image),
            "sample_path": str(destination_file),
        }

        if extra_metadata:
            row.update(extra_metadata)

        rows.append(row)


# ============================================================
# 1. GARBAGE DATASET
# ============================================================

print("\n" + "=" * 70)
print("GARBAGE DATASET")
print("=" * 70)

for label in GARBAGE_CLASSES:

    folder = GARBAGE / label

    images = get_images(folder)

    copy_samples(
        source="Garbage Dataset",
        label=label,
        images=images,
        destination=(
            OUTPUT
            / "garbage_dataset"
            / safe_name(label)
        ),
    )


# ============================================================
# 2. TRASHNET++
# ============================================================

print("\n" + "=" * 70)
print("TRASHNET++")
print("=" * 70)

trashnet_classwise = (
    TRASHNET
    / "Classwise Formet"
)

for label in TRASHNET_CLASSES:

    folder = (
        trashnet_classwise
        / label
    )

    images = get_images(folder)

    copy_samples(
        source="TrashNet++",
        label=label,
        images=images,
        destination=(
            OUTPUT
            / "trashnet_plus_plus"
            / safe_name(label)
        ),
    )


# ============================================================
# 3. MEDICAL WASTE 4.0
#
# IMPORTANT:
# DATASET A and DATASET B are kept separate.
#
# We do NOT merge them at this stage.
# ============================================================

print("\n" + "=" * 70)
print("MEDICAL WASTE 4.0")
print("=" * 70)

for dataset_split in [
    "DATASET A",
    "DATASET B",
]:

    split_root = MEDICAL / dataset_split

    print(f"\n--- {dataset_split} ---")

    for label in MEDICAL_CLASSES:

        folder = (
            split_root
            / label
        )

        images = get_images(folder)

        if not images:
            continue

        copy_samples(
            source="Medical Waste 4.0",
            label=label,
            images=images,
            destination=(
                OUTPUT
                / "medical_waste_4_0"
                / safe_name(dataset_split)
                / safe_name(label)
            ),
            extra_metadata={
                "medical_split": dataset_split
            },
        )


# ============================================================
# 4. WRITE MANIFEST
# ============================================================

manifest = (
    OUTPUT
    / "visual_overlap_sample_manifest.csv"
)

fieldnames = [
    "source",
    "label",
    "total_images",
    "sample_index",
    "original_path",
    "sample_path",
    "medical_split",
]

with open(
    manifest,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
        extrasaction="ignore",
    )

    writer.writeheader()
    writer.writerows(rows)


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 70)
print("VISUAL OVERLAP SAMPLING COMPLETE")
print("=" * 70)

print(f"\nSamples per class: {SAMPLES_PER_CLASS}")
print(f"Random seed: {SEED}")

print("\nTotal sampled images:", len(rows))

print("\nOutput:")
print(OUTPUT)

print("\nManifest:")
print(manifest)

print("\nIMPORTANT:")
print("Original datasets were NOT modified.")
print("No labels were changed.")
print("No datasets were merged.")
print("DATASET A and DATASET B remain separate.")
print("=" * 70)