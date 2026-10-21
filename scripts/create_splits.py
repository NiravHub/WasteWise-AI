from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# WasteWise AI - Stratified Dataset Split
# ============================================================

DATASET_DIR = Path("data/raw/garbage-dataset")
SPLIT_DIR = Path("data/splits")

RANDOM_SEED = 42

TRAIN_SIZE = 0.80
VALIDATION_SIZE = 0.10
TEST_SIZE = 0.10

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------

if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_DIR.resolve()}"
    )

if abs(TRAIN_SIZE + VALIDATION_SIZE + TEST_SIZE - 1.0) > 1e-9:
    raise ValueError("Split percentages must sum to 1.0")


SPLIT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Collect image paths and labels
# ------------------------------------------------------------

records = []

class_dirs = sorted(
    directory
    for directory in DATASET_DIR.iterdir()
    if directory.is_dir()
)


for class_dir in class_dirs:

    label = class_dir.name

    for image_path in class_dir.rglob("*"):

        if (
            image_path.is_file()
            and image_path.suffix.lower() in SUPPORTED_EXTENSIONS
        ):
            records.append(
                {
                    "path": str(image_path.resolve()),
                    "label": label,
                }
            )


df = pd.DataFrame(records)


if df.empty:
    raise RuntimeError("No images were found.")


print("=" * 70)
print("WASTEWISE AI - STRATIFIED DATASET SPLIT")
print("=" * 70)

print(f"\nTotal images: {len(df)}")

print("\nOriginal class distribution:")
print(
    df["label"]
    .value_counts()
    .sort_index()
)


# ------------------------------------------------------------
# First split: Train vs temporary
# ------------------------------------------------------------

train_df, temp_df = train_test_split(
    df,
    test_size=(VALIDATION_SIZE + TEST_SIZE),
    stratify=df["label"],
    random_state=RANDOM_SEED,
)


# ------------------------------------------------------------
# Second split: Validation vs Test
# ------------------------------------------------------------

relative_test_size = (
    TEST_SIZE /
    (VALIDATION_SIZE + TEST_SIZE)
)


validation_df, test_df = train_test_split(
    temp_df,
    test_size=relative_test_size,
    stratify=temp_df["label"],
    random_state=RANDOM_SEED,
)


# ------------------------------------------------------------
# Shuffle each split
# ------------------------------------------------------------

train_df = train_df.sample(
    frac=1,
    random_state=RANDOM_SEED,
).reset_index(drop=True)

validation_df = validation_df.sample(
    frac=1,
    random_state=RANDOM_SEED,
).reset_index(drop=True)

test_df = test_df.sample(
    frac=1,
    random_state=RANDOM_SEED,
).reset_index(drop=True)


# ------------------------------------------------------------
# Save CSV files
# ------------------------------------------------------------

train_path = SPLIT_DIR / "train.csv"
validation_path = SPLIT_DIR / "validation.csv"
test_path = SPLIT_DIR / "test.csv"

train_df.to_csv(
    train_path,
    index=False,
)

validation_df.to_csv(
    validation_path,
    index=False,
)

test_df.to_csv(
    test_path,
    index=False,
)


# ------------------------------------------------------------
# Display split statistics
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SPLIT SUMMARY")
print("=" * 70)

print(
    f"\nTraining   : {len(train_df):,} "
    f"({len(train_df) / len(df) * 100:.2f}%)"
)

print(
    f"Validation : {len(validation_df):,} "
    f"({len(validation_df) / len(df) * 100:.2f}%)"
)

print(
    f"Test       : {len(test_df):,} "
    f"({len(test_df) / len(df) * 100:.2f}%)"
)

print(
    f"Total      : "
    f"{len(train_df) + len(validation_df) + len(test_df):,}"
)


# ------------------------------------------------------------
# Class distributions
# ------------------------------------------------------------

for name, split_df in [
    ("TRAIN", train_df),
    ("VALIDATION", validation_df),
    ("TEST", test_df),
]:

    print("\n" + "-" * 70)
    print(f"{name} CLASS DISTRIBUTION")
    print("-" * 70)

    counts = (
        split_df["label"]
        .value_counts()
        .sort_index()
    )

    print(counts.to_string())


# ------------------------------------------------------------
# Verify no path overlap
# ------------------------------------------------------------

train_paths = set(train_df["path"])
validation_paths = set(validation_df["path"])
test_paths = set(test_df["path"])

train_val_overlap = train_paths & validation_paths
train_test_overlap = train_paths & test_paths
validation_test_overlap = validation_paths & test_paths

print("\n" + "=" * 70)
print("LEAKAGE CHECK")
print("=" * 70)

print(
    f"\nTrain ∩ Validation : {len(train_val_overlap)}"
)

print(
    f"Train ∩ Test       : {len(train_test_overlap)}"
)

print(
    f"Validation ∩ Test  : {len(validation_test_overlap)}"
)


if (
    train_val_overlap
    or train_test_overlap
    or validation_test_overlap
):
    raise RuntimeError(
        "DATA LEAKAGE DETECTED: "
        "the same image appears in multiple splits."
    )


# ------------------------------------------------------------
# Save class names
# ------------------------------------------------------------

class_names = sorted(df["label"].unique())

class_names_path = SPLIT_DIR / "class_names.txt"

class_names_path.write_text(
    "\n".join(class_names),
    encoding="utf-8",
)


# ------------------------------------------------------------
# Final status
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SPLIT COMPLETE")
print("=" * 70)

print("\nCreated:")

print(f"  {train_path}")
print(f"  {validation_path}")
print(f"  {test_path}")
print(f"  {class_names_path}")

print("\nDataset images were NOT copied or modified.")
print("Only split metadata CSV files were created.")