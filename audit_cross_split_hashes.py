from pathlib import Path
import hashlib

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

DATASETS = [
    Path("data/processed/taco_detection_practical"),
    Path("data/processed/garbage_external_wastewise"),
]


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def get_hashes(folder):
    files = [
        p for p in folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTS
    ]

    hashes = {}

    for i, path in enumerate(files, 1):
        h = sha256(path)
        hashes.setdefault(h, []).append(path)

        if i % 500 == 0:
            print(f"    hashed {i}/{len(files)}")

    return hashes


for dataset in DATASETS:

    print("\n" + "=" * 75)
    print("DATASET:", dataset)
    print("=" * 75)

    if not dataset.exists():
        print("SKIP: dataset does not exist")
        continue

    splits = {}

    for split in ["train", "val", "test"]:

        image_dir = dataset / "images" / split

        if not image_dir.exists():
            print(f"[{split.upper()}] not present")
            continue

        print(f"\n[{split.upper()}] {image_dir}")

        hashes = get_hashes(image_dir)

        splits[split] = hashes

        print(
            f"Images: {sum(len(v) for v in hashes.values())}"
        )

        print(
            f"Unique SHA256: {len(hashes)}"
        )

    split_names = list(splits.keys())

    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):

            a = split_names[i]
            b = split_names[j]

            overlap = set(splits[a]) & set(splits[b])

            print(
                f"\n{a.upper()} <-> {b.upper()} "
                f"exact SHA256 overlap: {len(overlap)}"
            )

            if overlap:

                print("!!! POSSIBLE EXACT LEAKAGE !!!")

                shown = 0

                for h in overlap:

                    print("\nSHA256:", h)

                    print("  " + a + ":")

                    for p in splits[a][h]:
                        print("    ", p)

                    print("  " + b + ":")

                    for p in splits[b][h]:
                        print("    ", p)

                    shown += 1

                    if shown >= 10:
                        print("\nShowing first 10 overlap groups.")
                        break


print("\n" + "=" * 75)
print("TACO / EXTERNAL EXACT LEAKAGE AUDIT COMPLETE")
print("=" * 75)