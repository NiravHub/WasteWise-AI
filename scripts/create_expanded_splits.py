from pathlib import Path
import hashlib
import json
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[1]

MANIFEST = ROOT / "results" / "dataset_audit" / "canonical_manifest.csv"
OUT = ROOT / "data" / "splits_expanded"
OUT.mkdir(parents=True, exist_ok=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    df = pd.read_csv(MANIFEST)

    print(f"Manifest records: {len(df):,}")

    # Clean training/validation/test pool
    eligible = df[
    df["policy"].isin([
        "core_and_expanded_train",
        "expanded_train",
        "medical_train_validation"
    ])
].copy()

    # Medical external test remains completely isolated
    medical_external = df[
        df["policy"] == "medical_external_test"
    ].copy()

    # Review set remains isolated
    review = df[
        df["policy"] == "expanded_train_review"
    ].copy()

    print(f"Eligible classification records: {len(eligible):,}")
    print(f"Medical external test: {len(medical_external):,}")
    print(f"Review set: {len(review):,}")

    # ---------------------------------------------------------
    # Compute SHA-256 groups
    # ---------------------------------------------------------
    print("\nComputing SHA-256 groups...")

    hashes = []

    for i, row in enumerate(eligible.itertuples(index=False), 1):
        path = Path(row.path)

        if not path.exists():
            raise FileNotFoundError(path)

        hashes.append(sha256_file(path))

        if i % 1000 == 0:
            print(f"  hashed {i:,}/{len(eligible):,}")

    eligible["sha256"] = hashes

    # ---------------------------------------------------------
    # Group-aware stratified split
    # ---------------------------------------------------------
    #
    # First split:
    #   80% train
    #   20% temporary
    #
    # Second split:
    #   50% validation
    #   50% test
    #
    # Final:
    #   80 / 10 / 10
    # ---------------------------------------------------------

    X = eligible.index
    y = eligible["canonical_class"]
    groups = eligible["sha256"]

    splitter1 = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    train_idx, temp_idx = next(
        splitter1.split(X, y, groups)
    )

    train = eligible.iloc[train_idx].copy()
    temp = eligible.iloc[temp_idx].copy()

    # 50/50 validation/test from temporary set
    X2 = temp.index
    y2 = temp["canonical_class"]
    groups2 = temp["sha256"]

    splitter2 = StratifiedGroupKFold(
        n_splits=2,
        shuffle=True,
        random_state=42
    )

    val_idx, test_idx = next(
        splitter2.split(X2, y2, groups2)
    )

    validation = temp.iloc[val_idx].copy()
    test = temp.iloc[test_idx].copy()

    # ---------------------------------------------------------
    # Safety checks
    # ---------------------------------------------------------

    train_hashes = set(train["sha256"])
    val_hashes = set(validation["sha256"])
    test_hashes = set(test["sha256"])

    assert train_hashes.isdisjoint(val_hashes)
    assert train_hashes.isdisjoint(test_hashes)
    assert val_hashes.isdisjoint(test_hashes)

    print("\nSHA-256 leakage checks: PASS")

    # ---------------------------------------------------------
    # Remove helper hash from public split CSVs?
    # Keep it — useful for reproducibility and leakage auditing.
    # ---------------------------------------------------------

    train.to_csv(OUT / "train.csv", index=False)
    validation.to_csv(OUT / "validation.csv", index=False)
    test.to_csv(OUT / "test.csv", index=False)

    medical_external.to_csv(
        OUT / "external_medical_test.csv",
        index=False
    )

    review.to_csv(
        OUT / "review.csv",
        index=False
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    summary = {
        "random_seed": 42,
        "total_eligible": len(eligible),
        "train": len(train),
        "validation": len(validation),
        "test": len(test),
        "medical_external_test": len(medical_external),
        "review": len(review),
        "split_ratios": {
            "train": len(train) / len(eligible),
            "validation": len(validation) / len(eligible),
            "test": len(test) / len(eligible)
        },
        "classes": sorted(
            eligible["canonical_class"].unique().tolist()
        ),
        "train_class_counts":
            train["canonical_class"].value_counts().sort_index().to_dict(),
        "validation_class_counts":
            validation["canonical_class"].value_counts().sort_index().to_dict(),
        "test_class_counts":
            test["canonical_class"].value_counts().sort_index().to_dict(),
        "sha256_group_leakage": False
    }

    with open(
        OUT / "split_summary.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("EXPANDED DATASET SPLIT COMPLETE")
    print("=" * 60)

    print(f"\nTrain:      {len(train):,}")
    print(f"Validation: {len(validation):,}")
    print(f"Test:       {len(test):,}")
    print(f"External:   {len(medical_external):,}")
    print(f"Review:     {len(review):,}")

    print("\nClass distribution:")
    print(
        pd.DataFrame({
            "train": train["canonical_class"].value_counts(),
            "validation": validation["canonical_class"].value_counts(),
            "test": test["canonical_class"].value_counts()
        })
        .fillna(0)
        .astype(int)
        .sort_index()
        .to_string()
    )

    print("\nSHA-256 leakage: PASS")
    print(f"\nSaved to: {OUT}")


if __name__ == "__main__":
    main()
