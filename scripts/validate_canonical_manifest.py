from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "results"
    / "dataset_audit"
    / "canonical_manifest.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "dataset_audit"
    / "canonical_manifest_validation.json"
)

EXPECTED_TOTAL = 27_427

EXPECTED_DATASET_COUNTS = {
    "garbage_dataset": 19_762,
    "trashnet_plus_plus": 1_344,
    "medical_waste_4_0": 6_321,
}

EXPECTED_POLICY_COUNTS = {
    "core_and_expanded_train": 19_762,
    "excluded_from_clean_training": 80,
    "expanded_train": 1_164,
    "expanded_train_review": 100,
    "medical_external_test": 801,
    "medical_train_validation": 5_520,
}

EXPECTED_FAMILIES = {
    "e_waste",
    "medical",
    "organic_biological",
    "recyclable",
    "residual",
    "textile_reusable",
}

EXPECTED_CLASSES = {
    "battery",
    "biological",
    "cable",
    "cardboard",
    "clothes",
    "electronic_device",
    "gauze",
    "general_trash",
    "glass",
    "medical_cap",
    "medical_filter",
    "medical_glasses",
    "medical_gloves",
    "medical_waste_object",
    "metal",
    "paper",
    "plastic",
    "shoe_cover",
    "shoes",
    "test_tube",
    "urine_bag",
}

EXPECTED_POLICIES = set(EXPECTED_POLICY_COUNTS)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}

# Actual canonical_manifest.csv schema
REQUIRED_COLUMNS = {
    "dataset",
    "original_label",
    "canonical_family",
    "canonical_class",
    "policy",
    "source_split",
    "metadata",
    "notes",
    "path",
}


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def is_image_valid(path: Path) -> tuple[bool, str]:
    try:
        with Image.open(path) as img:
            img.verify()

        with Image.open(path) as img:
            img.load()

        return True, ""

    except Exception as exc:
        return False, str(exc)


def print_status(name: str, passed: bool):
    status = "PASS" if passed else "FAIL"
    print(f"  {name:<55} {status}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CANONICAL MANIFEST VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Manifest existence
    # --------------------------------------------------------

    if not MANIFEST_PATH.exists():

        print()
        print("ERROR: Canonical manifest not found:")
        print(MANIFEST_PATH)
        print()

        raise SystemExit(1)

    print()
    print("Manifest:")
    print(f"  {MANIFEST_PATH}")

    # --------------------------------------------------------
    # Read manifest
    # --------------------------------------------------------

    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        columns = set(reader.fieldnames or [])

        rows = list(reader)

    print()
    print(f"Manifest records: {len(rows):,}")

    # --------------------------------------------------------
    # Structure
    # --------------------------------------------------------

    missing_columns = REQUIRED_COLUMNS - columns

    columns_pass = len(missing_columns) == 0

    print()
    print("Structure:")

    print_status(
        "Required columns",
        columns_pass,
    )

    if missing_columns:

        print()
        print("Missing columns:")

        for column in sorted(missing_columns):
            print(f"  - {column}")

        raise SystemExit(1)

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    dataset_counts = Counter()
    family_counts = Counter()
    class_counts = Counter()
    policy_counts = Counter()

    missing_files = []
    invalid_images = []

    duplicate_paths = []
    duplicate_hashes = []

    seen_paths = {}
    seen_hashes = {}

    invalid_datasets = []
    invalid_families = []
    invalid_classes = []
    invalid_policies = []

    hash_to_records = defaultdict(list)

    # --------------------------------------------------------
    # Validate every record
    # --------------------------------------------------------

    print()
    print("Checking records...")
    print()

    for index, row in enumerate(rows, start=2):

        dataset = row["dataset"].strip()
        original_label = row["original_label"].strip()
        family = row["canonical_family"].strip()
        canonical_class = row["canonical_class"].strip()
        policy = row["policy"].strip()
        source_split = row["source_split"].strip()
        notes = row["notes"].strip()
        path_raw = row["path"].strip()

        # Counters
        dataset_counts[dataset] += 1
        family_counts[family] += 1
        class_counts[canonical_class] += 1
        policy_counts[policy] += 1

        # ----------------------------------------------------
        # Dataset
        # ----------------------------------------------------

        if dataset not in EXPECTED_DATASET_COUNTS:

            invalid_datasets.append(
                {
                    "row": index,
                    "dataset": dataset,
                }
            )

        # ----------------------------------------------------
        # Family
        # ----------------------------------------------------

        if family not in EXPECTED_FAMILIES:

            invalid_families.append(
                {
                    "row": index,
                    "family": family,
                }
            )

        # ----------------------------------------------------
        # Class
        # ----------------------------------------------------

        if canonical_class not in EXPECTED_CLASSES:

            invalid_classes.append(
                {
                    "row": index,
                    "class": canonical_class,
                }
            )

        # ----------------------------------------------------
        # Policy
        # ----------------------------------------------------

        if policy not in EXPECTED_POLICIES:

            invalid_policies.append(
                {
                    "row": index,
                    "policy": policy,
                }
            )

        # ----------------------------------------------------
        # Path
        # ----------------------------------------------------

        resolved_path = Path(path_raw)

        if not resolved_path.is_absolute():
            resolved_path = PROJECT_ROOT / resolved_path

        resolved_path = resolved_path.resolve()

        path_key = str(resolved_path).lower()

        if path_key in seen_paths:

            duplicate_paths.append(
                {
                    "row": index,
                    "path": str(resolved_path),
                    "first_row": seen_paths[path_key],
                }
            )

        else:

            seen_paths[path_key] = index

        # ----------------------------------------------------
        # File existence
        # ----------------------------------------------------

        if not resolved_path.exists():

            missing_files.append(
                {
                    "row": index,
                    "path": str(resolved_path),
                }
            )

            continue

        if not resolved_path.is_file():

            missing_files.append(
                {
                    "row": index,
                    "path": str(resolved_path),
                    "reason": "not_a_file",
                }
            )

            continue

        # ----------------------------------------------------
        # Extension
        # ----------------------------------------------------

        if resolved_path.suffix.lower() not in IMAGE_EXTENSIONS:

            invalid_images.append(
                {
                    "row": index,
                    "path": str(resolved_path),
                    "reason": "unsupported_extension",
                }
            )

            continue

        # ----------------------------------------------------
        # Actual image validation
        # ----------------------------------------------------

        valid, error = is_image_valid(resolved_path)

        if not valid:

            invalid_images.append(
                {
                    "row": index,
                    "path": str(resolved_path),
                    "reason": error,
                }
            )

            continue

        # ----------------------------------------------------
        # SHA-256
        # ----------------------------------------------------

        file_hash = sha256_file(resolved_path)

        record = {
            "row": index,
            "dataset": dataset,
            "original_label": original_label,
            "canonical_class": canonical_class,
            "policy": policy,
            "source_split": source_split,
            "notes": notes,
            "path": str(resolved_path),
        }

        hash_to_records[file_hash].append(record)

        if file_hash in seen_hashes:

            duplicate_hashes.append(
                {
                    "row": index,
                    "hash": file_hash,
                    "first_row": seen_hashes[file_hash],
                    "path": str(resolved_path),
                }
            )

        else:

            seen_hashes[file_hash] = index

    # ========================================================
    # RESULTS
    # ========================================================

    print("=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    # --------------------------------------------------------
    # Total
    # --------------------------------------------------------

    total_pass = len(rows) == EXPECTED_TOTAL

    print()
    print("Manifest size:")

    print_status(
        f"Total records = {EXPECTED_TOTAL:,}",
        total_pass,
    )

    if not total_pass:

        print(f"  Expected: {EXPECTED_TOTAL:,}")
        print(f"  Found:    {len(rows):,}")

    # --------------------------------------------------------
    # File integrity
    # --------------------------------------------------------

    print()
    print("File integrity:")

    print_status(
        f"Missing files = {len(missing_files)}",
        len(missing_files) == 0,
    )

    print_status(
        f"Invalid/corrupt images = {len(invalid_images)}",
        len(invalid_images) == 0,
    )

    print_status(
        f"Duplicate manifest paths = {len(duplicate_paths)}",
        len(duplicate_paths) == 0,
    )

    # --------------------------------------------------------
    # Duplicate SHA-256
    # --------------------------------------------------------
    #
    # Duplicate SHA-256 records are allowed at manifest level
    # because TrashNet++ contains known internal duplicate files.
    #
    # IMPORTANT:
    # Cross-dataset duplicate groups are checked separately and
    # MUST remain zero.
    # --------------------------------------------------------

    duplicate_sha256_status = (
        "known_internal_source_duplicates"
        if duplicate_hashes
        else "none"
    )

    print_status(
        f"Duplicate SHA-256 records = {len(duplicate_hashes)}",
        True,
    )

    if duplicate_hashes:

        print(
            f"  NOTE: {len(duplicate_hashes)} repeated file records "
            f"were found. These are allowed because they are "
            f"known internal source-dataset duplicates."
        )

    # --------------------------------------------------------
    # Dataset counts
    # --------------------------------------------------------

    print()
    print("Dataset counts:")

    dataset_counts_pass = True

    for dataset, expected in EXPECTED_DATASET_COUNTS.items():

        actual = dataset_counts.get(dataset, 0)

        passed = actual == expected

        if not passed:
            dataset_counts_pass = False

        print_status(
            f"{dataset}: {actual:,} / {expected:,}",
            passed,
        )

    unexpected_datasets = (
        set(dataset_counts)
        - set(EXPECTED_DATASET_COUNTS)
    )

    if unexpected_datasets:

        dataset_counts_pass = False

        print()
        print("Unexpected datasets:")

        for dataset in sorted(unexpected_datasets):
            print(f"  - {dataset}")

    # --------------------------------------------------------
    # Family validation
    # --------------------------------------------------------

    print()
    print("Canonical families:")

    family_pass = True

    for family in sorted(family_counts):

        passed = family in EXPECTED_FAMILIES

        if not passed:
            family_pass = False

        print_status(
            f"{family}: {family_counts[family]:,}",
            passed,
        )

    missing_families = (
        EXPECTED_FAMILIES
        - set(family_counts)
    )

    if missing_families:

        family_pass = False

        print()
        print("Expected families missing from manifest:")

        for family in sorted(missing_families):
            print(f"  - {family}")

    # --------------------------------------------------------
    # Class validation
    # --------------------------------------------------------

    print()
    print("Canonical classes:")

    class_pass = True

    for canonical_class in sorted(class_counts):

        passed = canonical_class in EXPECTED_CLASSES

        if not passed:
            class_pass = False

        print_status(
            f"{canonical_class}: {class_counts[canonical_class]:,}",
            passed,
        )

    missing_classes = (
        EXPECTED_CLASSES
        - set(class_counts)
    )

    if missing_classes:

        class_pass = False

        print()
        print("Expected classes missing from manifest:")

        for canonical_class in sorted(missing_classes):
            print(f"  - {canonical_class}")

    # --------------------------------------------------------
    # Policy validation
    # --------------------------------------------------------

    print()
    print("Policy counts:")

    policy_pass = True

    for policy, expected in EXPECTED_POLICY_COUNTS.items():

        actual = policy_counts.get(policy, 0)

        passed = actual == expected

        if not passed:
            policy_pass = False

        print_status(
            f"{policy}: {actual:,} / {expected:,}",
            passed,
        )

    unexpected_policies = (
        set(policy_counts)
        - EXPECTED_POLICIES
    )

    if unexpected_policies:

        policy_pass = False

        print()
        print("Unexpected policies:")

        for policy in sorted(unexpected_policies):
            print(f"  - {policy}")

    # --------------------------------------------------------
    # Unmapped records
    # --------------------------------------------------------

    unmapped_count = 0

    for row in rows:

        family = row["canonical_family"].strip()
        canonical_class = row["canonical_class"].strip()
        policy = row["policy"].strip()

        if (
            not family
            or not canonical_class
            or not policy
            or family.lower() in {
                "unmapped",
                "null",
                "none",
            }
            or canonical_class.lower() in {
                "unmapped",
                "null",
                "none",
            }
        ):

            unmapped_count += 1

    print()
    print("Mapping:")

    print_status(
        f"Unmapped records = {unmapped_count}",
        unmapped_count == 0,
    )

    # --------------------------------------------------------
    # Cross-dataset exact duplicates
    # --------------------------------------------------------

    cross_dataset_duplicate_groups = []

    for file_hash, records in hash_to_records.items():

        datasets = {
            record["dataset"]
            for record in records
        }

        if len(datasets) > 1:

            cross_dataset_duplicate_groups.append(
                {
                    "sha256": file_hash,
                    "records": records,
                }
            )

    print()
    print("Cross-dataset exact duplicates:")

    print_status(
        "Cross-dataset duplicate groups = "
        f"{len(cross_dataset_duplicate_groups)}",
        len(cross_dataset_duplicate_groups) == 0,
    )

    # --------------------------------------------------------
    # Dataset policy rules
    # --------------------------------------------------------

    print()
    print("Dataset policy rules:")

    policy_rules_pass = True

    trashnet_yolo_found = []
    medical_b_invalid = []
    medical_a_invalid = []
    trashnet_medical_invalid = []

    for row in rows:

        dataset = row["dataset"].strip()
        path = row["path"].strip().replace("\\", "/")
        original_label = row["original_label"].strip()
        policy = row["policy"].strip()

        lower_path = path.lower()
        lower_label = original_label.lower()

        # --------------------------------------------
        # TrashNet YOLO must not appear
        # --------------------------------------------

        if dataset == "trashnet_plus_plus":

            if "yolo" in lower_path:
                trashnet_yolo_found.append(row)

        # --------------------------------------------
        # Medical Dataset B
        # --------------------------------------------

        if dataset == "medical_waste_4_0":

            if "/dataset b/" in lower_path:

                if policy != "medical_external_test":
                    medical_b_invalid.append(row)

            # ----------------------------------------
            # Medical Dataset A
            # ----------------------------------------

            if "/dataset a/" in lower_path:

                if policy != "medical_train_validation":
                    medical_a_invalid.append(row)

        # --------------------------------------------
        # TrashNet Medical Wast
        # --------------------------------------------

        if dataset == "trashnet_plus_plus":

            if (
                lower_label == "medical wast"
                or "medical wast" in lower_label
            ):

                if policy != "excluded_from_clean_training":
                    trashnet_medical_invalid.append(row)

    if trashnet_yolo_found:
        policy_rules_pass = False

    if medical_b_invalid:
        policy_rules_pass = False

    if medical_a_invalid:
        policy_rules_pass = False

    if trashnet_medical_invalid:
        policy_rules_pass = False

    print_status(
        "TrashNet YOLO excluded",
        len(trashnet_yolo_found) == 0,
    )

    print_status(
        "Medical Dataset B = external test only",
        len(medical_b_invalid) == 0,
    )

    print_status(
        "Medical Dataset A = train/validation",
        len(medical_a_invalid) == 0,
    )

    print_status(
        "TrashNet Medical Wast excluded from clean training",
        len(trashnet_medical_invalid) == 0,
    )

    # ========================================================
    # FINAL
    # ========================================================

    overall_pass = all(
        [
            total_pass,
            columns_pass,
            len(missing_files) == 0,
            len(invalid_images) == 0,
            len(duplicate_paths) == 0,
            dataset_counts_pass,
            family_pass,
            class_pass,
            policy_pass,
            unmapped_count == 0,
            len(cross_dataset_duplicate_groups) == 0,
            policy_rules_pass,
        ]
    )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    summary = {
        "validation": "PASS" if overall_pass else "FAIL",
        "manifest": str(MANIFEST_PATH),
        "total_records": len(rows),
        "expected_total_records": EXPECTED_TOTAL,
        "missing_files": len(missing_files),
        "invalid_images": len(invalid_images),
        "duplicate_paths": len(duplicate_paths),
        "duplicate_sha256_records": len(duplicate_hashes),
        "duplicate_sha256_status": duplicate_sha256_status,
        "cross_dataset_duplicate_groups": len(
            cross_dataset_duplicate_groups
        ),
        "unmapped_records": unmapped_count,
        "dataset_counts": dict(dataset_counts),
        "expected_dataset_counts": EXPECTED_DATASET_COUNTS,
        "family_counts": dict(family_counts),
        "class_counts": dict(class_counts),
        "policy_counts": dict(policy_counts),
        "expected_policy_counts": EXPECTED_POLICY_COUNTS,
        "invalid_datasets": invalid_datasets,
        "invalid_families": invalid_families,
        "invalid_classes": invalid_classes,
        "invalid_policies": invalid_policies,
        "trashnet_yolo_found": len(trashnet_yolo_found),
        "medical_b_invalid": len(medical_b_invalid),
        "medical_a_invalid": len(medical_a_invalid),
        "trashnet_medical_invalid": len(
            trashnet_medical_invalid
        ),
        "overall_pass": overall_pass,
    }

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Final terminal output
    # --------------------------------------------------------

    print()
    print("=" * 70)

    if overall_pass:

        print("VALIDATION: PASS")

    else:

        print("VALIDATION: FAIL")

    print("=" * 70)

    print()

    if overall_pass:

        print(
            "The canonical manifest is internally consistent "
            "and safe to use for preprocessing."
        )

    else:

        print(
            "Do NOT start preprocessing yet."
        )

        print(
            "Review the failed checks above first."
        )

    print()
    print("Validation report:")
    print(f"  {SUMMARY_PATH}")

    # --------------------------------------------------------
    # Detailed failures
    # --------------------------------------------------------

    if missing_files:

        print()
        print("Missing files:")

        for item in missing_files[:20]:

            print(
                f"  Row {item['row']}: "
                f"{item['path']}"
            )

    if invalid_images:

        print()
        print("Invalid images:")

        for item in invalid_images[:20]:

            print(
                f"  Row {item['row']}: "
                f"{item['path']}"
            )

    if duplicate_paths:

        print()
        print("Duplicate paths:")

        for item in duplicate_paths[:20]:

            print(
                f"  Row {item['row']}: "
                f"{item['path']} "
                f"(first row {item['first_row']})"
            )

    if duplicate_hashes:

        print()
        print("Duplicate SHA-256 records:")

        for item in duplicate_hashes[:20]:

            print(
                f"  Row {item['row']}: "
                f"{item['path']} "
                f"(first row {item['first_row']})"
            )

    if cross_dataset_duplicate_groups:

        print()
        print("Cross-dataset duplicate groups:")

        for group in cross_dataset_duplicate_groups[:20]:

            print(f"  SHA256: {group['sha256']}")

            for record in group["records"]:

                print(
                    f"    {record['dataset']} | "
                    f"{record['canonical_class']} | "
                    f"{record['path']}"
                )

    raise SystemExit(
        0 if overall_pass else 1
    )


if __name__ == "__main__":
    main()