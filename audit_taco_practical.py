from pathlib import Path
import hashlib

ROOT = Path("data/processed/taco_detection_practical")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def audit_split(split):
    folder = ROOT / "images" / split

    files = [
        p for p in folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTS
    ]

    hashes = {}

    print(f"\n[{split.upper()}]")
    print("Folder:", folder)
    print("Images found:", len(files))

    for i, path in enumerate(files, 1):
        h = sha256(path)
        hashes.setdefault(h, []).append(path)

        if i % 500 == 0:
            print(f"  hashed {i}/{len(files)}")

    duplicate_groups = {
        h: paths
        for h, paths in hashes.items()
        if len(paths) > 1
    }

    print("Unique SHA256:", len(hashes))
    print("Internal duplicate groups:", len(duplicate_groups))

    return hashes


print("=" * 75)
print("TACO PRACTICAL EXACT LEAKAGE AUDIT")
print("=" * 75)

train = audit_split("train")
val = audit_split("val")
test = audit_split("test")


def compare(name_a, hashes_a, name_b, hashes_b):

    overlap = set(hashes_a) & set(hashes_b)

    print(
        f"\n{name_a.upper()} <-> {name_b.upper()} "
        f"exact SHA256 overlap: {len(overlap)}"
    )

    if overlap:
        print("!!! EXACT CROSS-SPLIT LEAKAGE FOUND !!!")

        for h in list(overlap)[:10]:
            print("\nSHA256:", h)

            print(f"{name_a}:")
            for p in hashes_a[h]:
                print("  ", p)

            print(f"{name_b}:")
            for p in hashes_b[h]:
                print("  ", p)


compare("train", train, "val", val)
compare("train", train, "test", test)
compare("val", val, "test", test)


print("\n" + "=" * 75)
print("TACO PRACTICAL AUDIT COMPLETE")
print("=" * 75)