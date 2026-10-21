from pathlib import Path
from collections import defaultdict
import hashlib
import csv
import json
import math

from PIL import Image
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "garbage_dataset": (
        PROJECT_ROOT / "data" / "raw" / "garbage-dataset"
    ),
    "trashnet_plus_plus": (
        PROJECT_ROOT / "data" / "external" / "trashnet_plus_plus" / "TrashNet"
    ),
    "medical_waste_4_0": (
        PROJECT_ROOT
        / "data"
        / "external"
        / "medical_waste_4_0"
        / "Medical-Waste-4.0-Dataset"
        / "Medical Waste 4.0"
    ),
}

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "dataset_audit"
    / "cross_dataset_duplicates"
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}

# Hamming distance <= this value is considered a
# POTENTIAL visual duplicate and requires review.
PHASH_THRESHOLD = 2


# ============================================================
# UTILITIES
# ============================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def image_phash(path: Path) -> int | None:
    """
    Compute a simple 64-bit perceptual hash using
    an 8x8 low-frequency DCT representation.

    No external imagehash package is required.
    """

    try:
        with Image.open(path) as img:
            img = img.convert("L")
            img = img.resize((32, 32), Image.Resampling.LANCZOS)

            pixels = np.asarray(img, dtype=np.float32)

    except Exception:
        return None

    # 32x32 DCT transform matrix
    n = 32
    x = np.arange(n)

    alpha = np.ones(n)
    alpha[0] = 1 / math.sqrt(2)

    dct_matrix = np.cos(
        np.pi / n
        * (x[:, None] + 0.5)
        * x[None, :]
    )

    dct_matrix = (
        math.sqrt(2 / n)
        * alpha[:, None]
        * dct_matrix
    )

    dct = dct_matrix @ pixels @ dct_matrix.T

    # Take low-frequency 8x8 coefficients.
    low_freq = dct[:8, :8].flatten()

    # Exclude DC coefficient.
    median = np.median(low_freq[1:])

    bits = low_freq > median

    value = 0

    for bit in bits:
        value = (value << 1) | int(bit)

    return value


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def get_source_label(dataset_name: str, path: Path) -> str:
    """
    Preserve a useful source label based on the relative path.

    This does NOT modify the original dataset.
    """

    root = DATASETS[dataset_name]

    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = path

    parts = relative.parts

    if len(parts) == 1:
        return parts[0]

    return parts[0]


def collect_images():
    records = []

    print()
    print("=" * 70)
    print("COLLECTING IMAGE RECORDS")
    print("=" * 70)

    for dataset_name, root in DATASETS.items():

        if not root.exists():
            print(f"[WARNING] Missing: {root}")
            continue

        files = sorted(
            p
            for p in root.rglob("*")
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        )

        print(
            f"{dataset_name:25s} | "
            f"{len(files):6d} images | "
            f"{root}"
        )

        for path in files:
            records.append(
                {
                    "dataset": dataset_name,
                    "path": str(path),
                    "source_label": get_source_label(
                        dataset_name,
                        path
                    ),
                }
            )

    print()
    print(f"TOTAL IMAGE FILES: {len(records):,}")

    return records


# ============================================================
# EXACT DUPLICATE AUDIT
# ============================================================

def exact_duplicate_audit(records):

    print()
    print("=" * 70)
    print("LEVEL 1 — EXACT SHA-256 DUPLICATE AUDIT")
    print("=" * 70)

    hash_groups = defaultdict(list)

    for i, record in enumerate(records, start=1):

        path = Path(record["path"])

        try:
            digest = sha256_file(path)
        except Exception as exc:
            print(f"[ERROR] SHA-256 failed: {path}")
            print(f"        {exc}")
            continue

        record["sha256"] = digest

        hash_groups[digest].append(record)

        if i % 1000 == 0:
            print(f"Processed: {i:,}/{len(records):,}")

    cross_dataset_groups = []

    for digest, group in hash_groups.items():

        datasets = sorted(
            set(item["dataset"] for item in group)
        )

        if len(datasets) > 1:
            cross_dataset_groups.append(
                {
                    "sha256": digest,
                    "datasets": datasets,
                    "count": len(group),
                    "records": group,
                }
            )

    all_duplicate_groups = [
        {
            "sha256": digest,
            "datasets": sorted(
                set(item["dataset"] for item in group)
            ),
            "count": len(group),
            "records": group,
        }
        for digest, group in hash_groups.items()
        if len(group) > 1
    ]

    print()
    print(
        f"Unique SHA-256 values: "
        f"{len(hash_groups):,}"
    )

    print(
        f"Duplicate groups overall: "
        f"{len(all_duplicate_groups):,}"
    )

    print(
        f"Cross-dataset exact duplicate groups: "
        f"{len(cross_dataset_groups):,}"
    )

    return hash_groups, cross_dataset_groups


def save_exact_results(cross_dataset_groups):

    rows = []

    for group in cross_dataset_groups:

        for record in group["records"]:

            rows.append(
                {
                    "sha256": group["sha256"],
                    "dataset": record["dataset"],
                    "source_label": record["source_label"],
                    "path": record["path"],
                }
            )

    output = OUTPUT_DIR / "cross_dataset_exact_duplicates.csv"

    with output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sha256",
                "dataset",
                "source_label",
                "path",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    return output


# ============================================================
# PERCEPTUAL HASH
# ============================================================

def compute_phashes(records):

    print()
    print("=" * 70)
    print("LEVEL 2 — PERCEPTUAL HASH COMPUTATION")
    print("=" * 70)

    failed = 0

    for i, record in enumerate(records, start=1):

        path = Path(record["path"])

        phash = image_phash(path)

        if phash is None:
            failed += 1

        record["phash"] = phash

        if i % 1000 == 0:
            print(f"Processed: {i:,}/{len(records):,}")

    print()
    print(f"pHash failures: {failed}")

    return records


# ============================================================
# BK-TREE FOR HAMMING DISTANCE
# ============================================================

class BKTreeNode:

    def __init__(self, value, record):
        self.value = value
        self.records = [record]
        self.children = {}


class BKTree:

    def __init__(self):
        self.root = None

    def add(self, value, record):

        if self.root is None:
            self.root = BKTreeNode(value, record)
            return

        node = self.root

        while True:

            distance = hamming_distance(
                value,
                node.value
            )

            if distance == 0:
                node.records.append(record)
                return

            if distance not in node.children:

                node.children[distance] = BKTreeNode(
                    value,
                    record
                )

                return

            node = node.children[distance]

    def search(self, value, threshold):

        if self.root is None:
            return []

        results = []

        stack = [self.root]

        while stack:

            node = stack.pop()

            distance = hamming_distance(
                value,
                node.value
            )

            if distance <= threshold:

                for record in node.records:
                    results.append(
                        (
                            distance,
                            record
                        )
                    )

            lower = max(
                0,
                distance - threshold
            )

            upper = distance + threshold

            for edge, child in node.children.items():

                if lower <= edge <= upper:
                    stack.append(child)

        return results


def perceptual_duplicate_audit(records):

    print()
    print("=" * 70)
    print(
        "SEARCHING FOR POTENTIAL VISUAL DUPLICATES"
    )
    print("=" * 70)

    tree = BKTree()

    valid_records = [
        r
        for r in records
        if r.get("phash") is not None
    ]

    print(
        f"Valid pHash records: "
        f"{len(valid_records):,}"
    )

    for record in valid_records:

        tree.add(
            record["phash"],
            record
        )

    candidate_pairs = set()

    for i, record in enumerate(valid_records, start=1):

        matches = tree.search(
            record["phash"],
            PHASH_THRESHOLD
        )

        for distance, other in matches:

            if record["path"] == other["path"]:
                continue

            # Only cross-dataset candidates are relevant here.
            if record["dataset"] == other["dataset"]:
                continue

            paths = tuple(
                sorted(
                    [
                        record["path"],
                        other["path"],
                    ]
                )
            )

            candidate_pairs.add(
                (
                    paths[0],
                    paths[1],
                    distance,
                )
            )

        if i % 1000 == 0:
            print(
                f"Compared: "
                f"{i:,}/{len(valid_records):,}"
            )

    results = sorted(
        candidate_pairs,
        key=lambda x: x[2]
    )

    print()
    print(
        f"Cross-dataset perceptual candidates: "
        f"{len(results):,}"
    )

    return results


def save_perceptual_results(
    results,
    record_lookup
):

    output = (
        OUTPUT_DIR
        / "cross_dataset_perceptual_candidates.csv"
    )

    rows = []

    for path_a, path_b, distance in results:

        a = record_lookup[path_a]
        b = record_lookup[path_b]

        rows.append(
            {
                "hamming_distance": distance,
                "dataset_a": a["dataset"],
                "source_label_a": a["source_label"],
                "path_a": path_a,
                "dataset_b": b["dataset"],
                "source_label_b": b["source_label"],
                "path_b": path_b,
            }
        )

    with output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "hamming_distance",
                "dataset_a",
                "source_label_a",
                "path_a",
                "dataset_b",
                "source_label_b",
                "path_b",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    return output


# ============================================================
# SUMMARY
# ============================================================

def save_summary(
    records,
    exact_groups,
    perceptual_results,
):

    summary = {
        "datasets_audited": list(DATASETS.keys()),
        "total_image_files": len(records),
        "cross_dataset_exact_duplicate_groups": len(
            exact_groups
        ),
        "cross_dataset_perceptual_candidates": len(
            perceptual_results
        ),
        "perceptual_threshold_hamming": PHASH_THRESHOLD,
        "note": (
            "Perceptual candidates are not confirmed duplicates. "
            "They require visual review."
        ),
    }

    output = OUTPUT_DIR / "summary.json"

    with output.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    return output


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    records = collect_images()

    if not records:
        print("[ERROR] No images found.")
        return

    _, exact_groups = exact_duplicate_audit(
        records
    )

    exact_csv = save_exact_results(
        exact_groups
    )

    records = compute_phashes(records)

    perceptual_results = perceptual_duplicate_audit(
        records
    )

    record_lookup = {
        r["path"]: r
        for r in records
    }

    perceptual_csv = save_perceptual_results(
        perceptual_results,
        record_lookup,
    )

    summary = save_summary(
        records,
        exact_groups,
        perceptual_results,
    )

    print()
    print("=" * 70)
    print("CROSS-DATASET DUPLICATE AUDIT COMPLETE")
    print("=" * 70)

    print()
    print(f"Exact duplicates:")
    print(f"  {exact_csv}")

    print()
    print(f"Perceptual candidates:")
    print(f"  {perceptual_csv}")

    print()
    print(f"Summary:")
    print(f"  {summary}")

    print()
    print(
        "IMPORTANT: No source dataset was modified."
    )


if __name__ == "__main__":
    main()