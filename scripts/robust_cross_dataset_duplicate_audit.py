from pathlib import Path
from PIL import Image
import numpy as np
import hashlib
import csv
import json
import math
import time

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "garbage_dataset": ROOT / "data" / "raw" / "garbage-dataset",
    "trashnet_plus_plus": ROOT / "data" / "external" / "trashnet_plus_plus" / "TrashNet",
    "medical_waste_4_0": (
    ROOT
    / "data"
    / "external"
    / "medical_waste_4_0"
    / "Medical-Waste-4.0-Dataset"
    / "Medical Waste 4.0"
),
}

OUT = ROOT / "results" / "dataset_audit" / "robust_duplicate_audit"
OUT.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

print("=" * 70)
print("ROBUST CROSS-DATASET DUPLICATE AUDIT")
print("=" * 70)


# ---------------------------------------------------------------------
# SHA-256
# ---------------------------------------------------------------------

def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)

    return h.hexdigest()


# ---------------------------------------------------------------------
# Source/class extraction
# ---------------------------------------------------------------------

def get_source_label(dataset_name, path, root):
    rel = path.relative_to(root)
    parts = rel.parts

    if dataset_name == "garbage_dataset":
        return parts[0] if parts else "unknown"

    if dataset_name == "medical_waste_4_0":
        # DATASET A / class / image
        # DATASET B / class / image
        if len(parts) >= 2:
            return f"{parts[0]} / {parts[1]}"
        return parts[0] if parts else "unknown"

    if dataset_name == "trashnet_plus_plus":
        # Preserve enough hierarchy to distinguish Classwise/YOLO.
        if len(parts) >= 2:
            return f"{parts[0]} / {parts[1]}"
        return parts[0] if parts else "unknown"

    return parts[0] if parts else "unknown"


# ---------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------

def load_gray(path, size=32):
    try:
        with Image.open(path) as img:
            img = img.convert("L")
            img = img.resize((size, size), Image.Resampling.LANCZOS)
            return np.asarray(img, dtype=np.float32)
    except Exception:
        return None


def load_rgb(path, size=64):
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            img = img.resize((size, size), Image.Resampling.LANCZOS)
            return np.asarray(img, dtype=np.float32)
    except Exception:
        return None


# ---------------------------------------------------------------------
# DCT
# ---------------------------------------------------------------------

def dct_matrix(n):
    matrix = np.zeros((n, n), dtype=np.float64)

    for k in range(n):
        for i in range(n):
            if k == 0:
                alpha = math.sqrt(1 / n)
            else:
                alpha = math.sqrt(2 / n)

            matrix[k, i] = (
                alpha *
                math.cos(math.pi * (2 * i + 1) * k / (2 * n))
            )

    return matrix


DCT32 = dct_matrix(32)


# ---------------------------------------------------------------------
# 256-bit pHash
# ---------------------------------------------------------------------

def phash256(gray):
    normalized = gray / 255.0

    dct = DCT32 @ normalized @ DCT32.T

    low = dct[:16, :16]

    values = low.flatten()

    # Exclude DC coefficient from threshold calculation.
    median = np.median(values[1:])

    bits = values > median

    result = 0

    for bit in bits:
        result = (result << 1) | int(bit)

    return result


# ---------------------------------------------------------------------
# 256-bit dHash
# ---------------------------------------------------------------------

def dhash256(gray):
    # 17 x 16 gives 16 horizontal comparisons across 16 rows.
    img = Image.fromarray(gray.astype(np.uint8))
    img = img.resize((17, 16), Image.Resampling.LANCZOS)

    arr = np.asarray(img, dtype=np.uint8)

    diff = arr[:, 1:] > arr[:, :-1]

    bits = diff.flatten()

    result = 0

    for bit in bits:
        result = (result << 1) | int(bit)

    return result


# ---------------------------------------------------------------------
# Hamming distance
# ---------------------------------------------------------------------

def hamming(a, b):
    return (a ^ b).bit_count()


# ---------------------------------------------------------------------
# Pixel verification
# ---------------------------------------------------------------------

def pixel_similarity(path1, path2):
    a = load_rgb(path1, 64)
    b = load_rgb(path2, 64)

    if a is None or b is None:
        return None

    # Normalize brightness slightly so JPEG/compression changes
    # don't dominate the comparison.
    a = a / 255.0
    b = b / 255.0

    mae = np.mean(np.abs(a - b))

    # Convert MAE into an intuitive similarity score.
    similarity = max(0.0, 1.0 - mae)

    return float(similarity)


# ---------------------------------------------------------------------
# Collect files
# ---------------------------------------------------------------------

records = []

print("\nCollecting unique image contents...")

for dataset_name, root in DATASETS.items():

    if not root.exists():
        print(f"[WARNING] Missing: {root}")
        continue

    files = [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    print(f"{dataset_name:25s}: {len(files):,} files")

    seen_sha = set()

    for path in files:
        try:
            sha = sha256_file(path)
        except Exception:
            continue

        # Remove internal exact duplicates from candidate generation.
        if sha in seen_sha:
            continue

        seen_sha.add(sha)

        records.append({
            "dataset": dataset_name,
            "path": str(path),
            "source_label": get_source_label(dataset_name, path, root),
            "sha256": sha,
        })

print(f"\nUnique image contents: {len(records):,}")


# ---------------------------------------------------------------------
# Save manifest
# ---------------------------------------------------------------------

manifest_path = OUT / "unique_image_manifest.csv"

with open(manifest_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["dataset", "source_label", "sha256", "path"]
    )
    writer.writeheader()
    writer.writerows(records)

print(f"Manifest saved: {manifest_path}")


# ---------------------------------------------------------------------
# Compute perceptual signatures
# ---------------------------------------------------------------------

print("\nComputing 256-bit pHash + dHash signatures...")

valid_records = []

start = time.time()

for i, record in enumerate(records, 1):

    path = Path(record["path"])

    gray = load_gray(path)

    if gray is None:
        continue

    try:
        ph = phash256(gray)
        dh = dhash256(gray)
    except Exception:
        continue

    record["phash"] = ph
    record["dhash"] = dh

    valid_records.append(record)

    if i % 1000 == 0:
        print(f"  processed {i:,}/{len(records):,}")

print(
    f"Signature generation completed in "
    f"{time.time() - start:.1f} seconds."
)

# ---------------------------------------------------------------------
# Candidate generation
#
# Important:
# We use BOTH hashes.
#
# This intentionally sacrifices recall for precision.
# ---------------------------------------------------------------------

PHASH_THRESHOLD = 8
DHASH_THRESHOLD = 8

print("\nGenerating cross-dataset candidates...")
print(f"pHash threshold: {PHASH_THRESHOLD}")
print(f"dHash threshold: {DHASH_THRESHOLD}")

# Group records by dataset.
by_dataset = {}

for r in valid_records:
    by_dataset.setdefault(r["dataset"], []).append(r)

dataset_names = list(by_dataset.keys())

candidates = []

pair_counts = {}

for i in range(len(dataset_names)):
    for j in range(i + 1, len(dataset_names)):

        d1 = dataset_names[i]
        d2 = dataset_names[j]

        group1 = by_dataset[d1]
        group2 = by_dataset[d2]

        pair_key = f"{d1} <-> {d2}"
        count = 0

        print(f"\nComparing {d1} ↔ {d2}")
        print(f"  {len(group1):,} × {len(group2):,}")

        # This is deliberately simple and deterministic.
        # 28k images is manageable for this audit.
        for a in group1:

            for b in group2:

                pd = hamming(a["phash"], b["phash"])

                if pd > PHASH_THRESHOLD:
                    continue

                dd = hamming(a["dhash"], b["dhash"])

                if dd > DHASH_THRESHOLD:
                    continue

                candidates.append({
                    "dataset_a": d1,
                    "dataset_b": d2,
                    "source_label_a": a["source_label"],
                    "source_label_b": b["source_label"],
                    "path_a": a["path"],
                    "path_b": b["path"],
                    "sha256_a": a["sha256"],
                    "sha256_b": b["sha256"],
                    "phash_distance": pd,
                    "dhash_distance": dd,
                })

                count += 1

        pair_counts[pair_key] = count

        print(f"  candidates: {count:,}")


print("\nTotal perceptual candidates:", len(candidates))


# ---------------------------------------------------------------------
# Pixel verification
# ---------------------------------------------------------------------

print("\nVerifying candidates using pixel-level similarity...")

confirmed = []
probable = []
similar = []

for idx, candidate in enumerate(candidates, 1):

    sim = pixel_similarity(
        Path(candidate["path_a"]),
        Path(candidate["path_b"])
    )

    candidate["pixel_similarity"] = sim

    if sim is None:
        continue

    # These thresholds are intentionally conservative.
    if sim >= 0.985:
        candidate["verification"] = "confirmed_duplicate"
        confirmed.append(candidate)

    elif sim >= 0.95:
        candidate["verification"] = "probable_duplicate"
        probable.append(candidate)

    else:
        candidate["verification"] = "visually_similar"
        similar.append(candidate)

    if idx % 1000 == 0:
        print(f"  verified {idx:,}/{len(candidates):,}")


# ---------------------------------------------------------------------
# Save candidate CSV
# ---------------------------------------------------------------------

candidate_path = OUT / "perceptual_candidates.csv"

fieldnames = [
    "dataset_a",
    "dataset_b",
    "source_label_a",
    "source_label_b",
    "path_a",
    "path_b",
    "sha256_a",
    "sha256_b",
    "phash_distance",
    "dhash_distance",
    "pixel_similarity",
    "verification",
]

with open(candidate_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for row in candidates:
        writer.writerow(row)


# ---------------------------------------------------------------------
# Save confirmed duplicates
# ---------------------------------------------------------------------

confirmed_path = OUT / "confirmed_cross_dataset_duplicates.csv"

with open(confirmed_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for row in confirmed:
        writer.writerow(row)


# ---------------------------------------------------------------------
# Save probable duplicates
# ---------------------------------------------------------------------

probable_path = OUT / "probable_cross_dataset_duplicates.csv"

with open(probable_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for row in probable:
        writer.writerow(row)


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

summary = {
    "unique_image_contents_compared": len(valid_records),
    "perceptual_candidates": len(candidates),
    "confirmed_cross_dataset_duplicates": len(confirmed),
    "probable_cross_dataset_duplicates": len(probable),
    "visually_similar_candidates": len(similar),
    "phash_threshold": PHASH_THRESHOLD,
    "dhash_threshold": DHASH_THRESHOLD,
    "pixel_confirm_threshold": 0.985,
    "pixel_probable_threshold": 0.95,
    "dataset_pair_candidate_counts": pair_counts,
}

summary_path = OUT / "robust_duplicate_audit_summary.json"

with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)


# ---------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------

print("\n")
print("=" * 70)
print("ROBUST DUPLICATE AUDIT COMPLETE")
print("=" * 70)

print(f"Unique image contents:       {len(valid_records):,}")
print(f"Perceptual candidates:       {len(candidates):,}")
print(f"Confirmed duplicates:        {len(confirmed):,}")
print(f"Probable duplicates:         {len(probable):,}")
print(f"Visually similar:            {len(similar):,}")

print("\nOutput:")
print(f"  {manifest_path}")
print(f"  {candidate_path}")
print(f"  {confirmed_path}")
print(f"  {probable_path}")
print(f"  {summary_path}")

print("\nIMPORTANT:")
print("No original dataset files were modified.")
print("No images were deleted or moved.")
print("=" * 70)