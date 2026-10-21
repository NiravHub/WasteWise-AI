from pathlib import Path
import hashlib
from PIL import Image
import imagehash

ROOT = Path(__file__).resolve().parents[1]

TACO_DIR = ROOT / "data" / "processed" / "taco_detection_practical" / "images"
EXT_DIR = ROOT / "data" / "external" / "garbage_object_detection"

OUTPUT = ROOT / "results" / "dataset_audit" / "taco_external_duplicate_audit"
OUTPUT.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(root):
    return [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def phash(path):
    with Image.open(path) as im:
        return imagehash.phash(im.convert("RGB"))


print("=" * 70)
print("TACO ↔ EXTERNAL DATASET DUPLICATE AUDIT")
print("=" * 70)

taco = collect_images(TACO_DIR)
external = collect_images(EXT_DIR)

print(f"TACO images:      {len(taco)}")
print(f"External images:  {len(external)}")

# ------------------------------------------------------------
# EXACT SHA-256 AUDIT
# ------------------------------------------------------------

print("\n[1/2] Computing SHA-256 hashes...")

taco_sha = {}
external_sha = {}

for i, path in enumerate(taco, 1):
    taco_sha.setdefault(sha256(path), []).append(path)

    if i % 250 == 0:
        print(f"TACO: {i}/{len(taco)}")

for i, path in enumerate(external, 1):
    external_sha.setdefault(sha256(path), []).append(path)

    if i % 1000 == 0:
        print(f"External: {i}/{len(external)}")

exact_matches = []

for digest, taco_paths in taco_sha.items():
    if digest in external_sha:
        for t in taco_paths:
            for e in external_sha[digest]:
                exact_matches.append((t, e, digest))

print(f"\nExact cross-dataset matches: {len(exact_matches)}")

with open(OUTPUT / "exact_matches.txt", "w", encoding="utf-8") as f:
    for t, e, digest in exact_matches:
        f.write(f"TACO\t{t}\n")
        f.write(f"EXT\t{e}\n")
        f.write(f"SHA256\t{digest}\n\n")

# ------------------------------------------------------------
# PERCEPTUAL HASH AUDIT
# ------------------------------------------------------------

print("\n[2/2] Computing perceptual hashes...")

taco_phash = {}

for i, path in enumerate(taco, 1):
    try:
        taco_phash[path] = phash(path)
    except Exception as exc:
        print(f"TACO hash failed: {path} -> {exc}")

    if i % 250 == 0:
        print(f"TACO pHash: {i}/{len(taco)}")

external_phash = {}

for i, path in enumerate(external, 1):
    try:
        external_phash[path] = phash(path)
    except Exception as exc:
        print(f"External hash failed: {path} -> {exc}")

    if i % 1000 == 0:
        print(f"External pHash: {i}/{len(external)}")


# pHash distance <= 8 is a candidate.
# We do NOT automatically call these duplicates.
candidates = []

for t_path, t_hash in taco_phash.items():
    for e_path, e_hash in external_phash.items():
        distance = t_hash - e_hash

        if distance <= 8:
            candidates.append(
                (distance, t_path, e_path)
            )

candidates.sort(key=lambda x: x[0])

print(f"\nPerceptual candidates (pHash distance <= 8): {len(candidates)}")

with open(OUTPUT / "perceptual_candidates.txt", "w", encoding="utf-8") as f:
    for distance, t, e in candidates:
        f.write(f"distance={distance}\n")
        f.write(f"TACO\t{t}\n")
        f.write(f"EXT\t{e}\n\n")


# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

summary = f"""
TACO ↔ EXTERNAL DUPLICATE AUDIT

TACO images: {len(taco)}
External images: {len(external)}

Exact SHA-256 cross-dataset matches: {len(exact_matches)}
Perceptual candidates (pHash <= 8): {len(candidates)}

Interpretation:
- Exact matches are definite duplicate-content leakage.
- Perceptual matches are candidates requiring visual inspection.
- No dataset files were modified.
"""

print(summary)

with open(OUTPUT / "summary.txt", "w", encoding="utf-8") as f:
    f.write(summary)

print(f"Results saved to:")
print(OUTPUT)