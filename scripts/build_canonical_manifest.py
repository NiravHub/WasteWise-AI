from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parents[1]

OUT_DIR = ROOT / "results" / "dataset_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "canonical_manifest.csv"
OUT_JSON = OUT_DIR / "canonical_manifest_summary.json"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"
}

# ---------------------------------------------------------------------
# Dataset roots
# ---------------------------------------------------------------------

GARBAGE_ROOT = ROOT / "data" / "raw" / "garbage-dataset"

TRASHNET_ROOT = (
    ROOT / "data" / "external" / "trashnet_plus_plus" / "TrashNet"
)

MEDICAL_ROOT = (
    ROOT
    / "data"
    / "external"
    / "medical_waste_4_0"
    / "Medical-Waste-4.0-Dataset"
    / "Medical Waste 4.0"
)

# ---------------------------------------------------------------------
# Garbage Dataset
# ---------------------------------------------------------------------

GARBAGE_MAP = {
    "battery": {
        "family": "e_waste",
        "canonical_class": "battery",
        "policy": "core_and_expanded_train",
    },
    "biological": {
        "family": "organic_biological",
        "canonical_class": "biological",
        "policy": "core_and_expanded_train",
    },
    "cardboard": {
        "family": "recyclable",
        "canonical_class": "cardboard",
        "policy": "core_and_expanded_train",
    },
    "clothes": {
        "family": "textile_reusable",
        "canonical_class": "clothes",
        "policy": "core_and_expanded_train",
    },
    "glass": {
        "family": "recyclable",
        "canonical_class": "glass",
        "policy": "core_and_expanded_train",
    },
    "metal": {
        "family": "recyclable",
        "canonical_class": "metal",
        "policy": "core_and_expanded_train",
    },
    "paper": {
        "family": "recyclable",
        "canonical_class": "paper",
        "policy": "core_and_expanded_train",
    },
    "plastic": {
        "family": "recyclable",
        "canonical_class": "plastic",
        "policy": "core_and_expanded_train",
    },
    "shoes": {
        "family": "textile_reusable",
        "canonical_class": "shoes",
        "policy": "core_and_expanded_train",
    },
    "trash": {
        "family": "residual",
        "canonical_class": "general_trash",
        "policy": "core_and_expanded_train",
    },
}

# ---------------------------------------------------------------------
# TrashNet++
#
# "Medical Wast" is deliberately excluded from clean medical training.
# ---------------------------------------------------------------------

TRASHNET_MAP = {
    "Plastic": {
        "family": "recyclable",
        "canonical_class": "plastic",
        "policy": "expanded_train",
        "notes": "additional_source",
    },
    "Glass": {
        "family": "recyclable",
        "canonical_class": "glass",
        "policy": "expanded_train",
        "notes": "additional_source",
    },
    "Metal": {
        "family": "recyclable",
        "canonical_class": "metal",
        "policy": "expanded_train",
        "notes": "additional_source",
    },
    "Can": {
        "family": "recyclable",
        "canonical_class": "metal",
        "policy": "expanded_train_review",
        "notes": "metal_packaging; preserve_original_label",
    },
    "E-wast": {
        "family": "e_waste",
        "canonical_class": "electronic_device",
        "policy": "expanded_train",
        "notes": "additional_source",
    },
    "Cable": {
        "family": "e_waste",
        "canonical_class": "cable",
        "policy": "expanded_train",
        "notes": "additional_source",
    },
    "Medical Wast": {
        "family": "medical",
        "canonical_class": "medical_waste_object",
        "policy": "excluded_from_clean_training",
        "notes": "noisy_broad_label",
    },
}

# ---------------------------------------------------------------------
# Medical Waste 4.0
# ---------------------------------------------------------------------

MEDICAL_CLASS_MAP = {
    "gauze": {
        "canonical_class": "gauze",
        "metadata": {"material": "unspecified"},
    },
    "gauzes": {
        "canonical_class": "gauze",
        "metadata": {"material": "unspecified"},
    },

    "glove_pair_latex": {
        "canonical_class": "medical_gloves",
        "metadata": {"material": "latex", "quantity": "pair"},
    },
    "glove_pair_nitrile": {
        "canonical_class": "medical_gloves",
        "metadata": {"material": "nitrile", "quantity": "pair"},
    },
    "glove_pair_surgery": {
        "canonical_class": "medical_gloves",
        "metadata": {"material": "surgical", "quantity": "pair"},
    },

    "glove_single_latex": {
        "canonical_class": "medical_gloves",
        "metadata": {"material": "latex", "quantity": "single"},
    },
    "glove_single_nitrile": {
        "canonical_class": "medical_gloves",
        "metadata": {"material": "nitrile", "quantity": "single"},
    },
    "glove_single_surgery": {
        "canonical_class": "medical_gloves",
        "metadata": {"material": "surgical", "quantity": "single"},
    },

    "medical_cap": {
        "canonical_class": "medical_cap",
        "metadata": {},
    },
    "medical_filter": {
        "canonical_class": "medical_filter",
        "metadata": {},
    },
    "medical_glasses": {
        "canonical_class": "medical_glasses",
        "metadata": {},
    },

    "shoe_cover_pair": {
        "canonical_class": "shoe_cover",
        "metadata": {"quantity": "pair"},
    },
    "shoe_cover_single": {
        "canonical_class": "shoe_cover",
        "metadata": {"quantity": "single"},
    },

    "test_tube": {
        "canonical_class": "test_tube",
        "metadata": {},
    },
    "urine_bag": {
        "canonical_class": "urine_bag",
        "metadata": {},
    },
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def make_record(
    dataset,
    path,
    original_label,
    family,
    canonical_class,
    policy,
    source_split="",
    metadata=None,
    notes="",
):
    return {
        "dataset": dataset,
        "original_label": original_label,
        "canonical_family": family,
        "canonical_class": canonical_class,
        "policy": policy,
        "source_split": source_split,
        "metadata": json.dumps(metadata or {}, sort_keys=True),
        "notes": notes,
        "path": str(path),
    }


records = []
unmapped = []


# ---------------------------------------------------------------------
# Garbage Dataset
# ---------------------------------------------------------------------

print("=" * 70)
print("BUILDING CANONICAL MANIFEST")
print("=" * 70)

print("\nGarbage Dataset...")

garbage_count = 0

if GARBAGE_ROOT.exists():

    for path in GARBAGE_ROOT.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        rel = path.relative_to(GARBAGE_ROOT)
        parts = rel.parts

        if len(parts) < 2:
            unmapped.append({
                "dataset": "garbage_dataset",
                "path": str(path),
                "reason": "missing_class_directory",
            })
            continue

        original_label = parts[0]

        mapping = GARBAGE_MAP.get(original_label)

        if mapping is None:
            unmapped.append({
                "dataset": "garbage_dataset",
                "path": str(path),
                "reason": f"unknown_label:{original_label}",
            })
            continue

        records.append(
            make_record(
                dataset="garbage_dataset",
                path=path,
                original_label=original_label,
                family=mapping["family"],
                canonical_class=mapping["canonical_class"],
                policy=mapping["policy"],
                source_split="",
                notes="original_core_dataset",
            )
        )

        garbage_count += 1

print(f"Mapped: {garbage_count:,}")


# ---------------------------------------------------------------------
# TrashNet++
# ---------------------------------------------------------------------

print("\nTrashNet++...")

trashnet_count = 0

if TRASHNET_ROOT.exists():

    for path in TRASHNET_ROOT.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        rel = path.relative_to(TRASHNET_ROOT)
        parts = rel.parts

        # We only use Classwise Formet for clean class-level mapping.
        #
        # YOLO Formet is detection-format material and is not treated
        # as an independent classification source here.
        if len(parts) >= 3 and parts[0].lower() == "classwise formet".lower():

            original_label = parts[1]

            mapping = TRASHNET_MAP.get(original_label)

            if mapping is None:
                unmapped.append({
                    "dataset": "trashnet_plus_plus",
                    "path": str(path),
                    "reason": f"unknown_classwise_label:{original_label}",
                })
                continue

            records.append(
                make_record(
                    dataset="trashnet_plus_plus",
                    path=path,
                    original_label=original_label,
                    family=mapping["family"],
                    canonical_class=mapping["canonical_class"],
                    policy=mapping["policy"],
                    source_split="classwise",
                    notes=mapping.get("notes", ""),
                )
            )

            trashnet_count += 1

        # Everything outside Classwise Formet is intentionally ignored.
        # This prevents the YOLO representation from being double-counted.

print(f"Mapped Classwise images: {trashnet_count:,}")


# ---------------------------------------------------------------------
# Medical Waste 4.0
# ---------------------------------------------------------------------

print("\nMedical Waste 4.0...")

medical_count = 0

if MEDICAL_ROOT.exists():

    for dataset_dir in sorted(MEDICAL_ROOT.iterdir()):

        if not dataset_dir.is_dir():
            continue

        dataset_name = dataset_dir.name

        if dataset_name not in {"DATASET A", "DATASET B"}:
            continue

        if dataset_name == "DATASET A":
            policy = "medical_train_validation"
        else:
            policy = "medical_external_test"

        for class_dir in sorted(dataset_dir.iterdir()):

            if not class_dir.is_dir():
                continue

            original_label = class_dir.name

            mapping = MEDICAL_CLASS_MAP.get(original_label)

            if mapping is None:
                unmapped.append({
                    "dataset": "medical_waste_4_0",
                    "path": str(class_dir),
                    "reason": f"unknown_medical_label:{original_label}",
                })
                continue

            for path in class_dir.rglob("*"):

                if not path.is_file():
                    continue

                if path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                records.append(
                    make_record(
                        dataset="medical_waste_4_0",
                        path=path,
                        original_label=original_label,
                        family="medical",
                        canonical_class=mapping["canonical_class"],
                        policy=policy,
                        source_split=dataset_name,
                        metadata=mapping.get("metadata", {}),
                        notes=(
                            "Dataset A training/validation source"
                            if dataset_name == "DATASET A"
                            else "Dataset B external test source"
                        ),
                    )
                )

                medical_count += 1

print(f"Mapped medical images: {medical_count:,}")


# ---------------------------------------------------------------------
# Save manifest
# ---------------------------------------------------------------------

fieldnames = [
    "dataset",
    "original_label",
    "canonical_family",
    "canonical_class",
    "policy",
    "source_split",
    "metadata",
    "notes",
    "path",
]

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:

    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(records)


# ---------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------

def counts(field):
    result = {}

    for r in records:
        key = r[field]
        result[key] = result.get(key, 0) + 1

    return dict(sorted(result.items()))


summary = {
    "total_manifest_records": len(records),
    "unmapped_records": len(unmapped),
    "dataset_counts": counts("dataset"),
    "family_counts": counts("canonical_family"),
    "canonical_class_counts": counts("canonical_class"),
    "policy_counts": counts("policy"),
    "unmapped": unmapped,
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)


# ---------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("CANONICAL MANIFEST SUMMARY")
print("=" * 70)

print(f"\nTotal manifest records: {len(records):,}")
print(f"Unmapped records:       {len(unmapped):,}")

print("\nDataset counts:")
for k, v in summary["dataset_counts"].items():
    print(f"  {k:30s} {v:8,}")

print("\nCanonical family counts:")
for k, v in summary["family_counts"].items():
    print(f"  {k:30s} {v:8,}")

print("\nCanonical class counts:")
for k, v in summary["canonical_class_counts"].items():
    print(f"  {k:30s} {v:8,}")

print("\nPolicy counts:")
for k, v in summary["policy_counts"].items():
    print(f"  {k:35s} {v:8,}")

print("\nOutput:")
print(f"  {OUT_CSV}")
print(f"  {OUT_JSON}")

print("\n" + "=" * 70)

if unmapped:
    print("WARNING: UNMAPPED RECORDS EXIST")
    print("Do NOT proceed to preprocessing until they are reviewed.")
else:
    print("ALL DISCOVERED CLASSIFICATION IMAGES ARE MAPPED.")
    print("Safe to proceed to manifest validation.")

print("=" * 70)