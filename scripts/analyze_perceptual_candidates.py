from pathlib import Path
import pandas as pd
import json

ROOT = Path("results/dataset_audit/cross_dataset_duplicates")
CSV_PATH = ROOT / "cross_dataset_perceptual_candidates.csv"

print("=" * 70)
print("CROSS-DATASET PERCEPTUAL CANDIDATE ANALYSIS")
print("=" * 70)

df = pd.read_csv(CSV_PATH)

print(f"\nTotal candidate pairs: {len(df):,}")

# ------------------------------------------------------------
# Hamming distance distribution
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("1. HAMMING DISTANCE DISTRIBUTION")
print("=" * 70)

distance_counts = (
    df["hamming_distance"]
    .value_counts()
    .sort_index()
)

for distance, count in distance_counts.items():
    pct = count / len(df) * 100
    print(f"Distance {distance}: {count:,} ({pct:.2f}%)")

# ------------------------------------------------------------
# Dataset pair distribution
# ------------------------------------------------------------

df["dataset_pair"] = df.apply(
    lambda r: " ↔ ".join(
        sorted([str(r["dataset_a"]), str(r["dataset_b"])])
    ),
    axis=1,
)

print("\n" + "=" * 70)
print("2. DATASET-PAIR DISTRIBUTION")
print("=" * 70)

pair_counts = df["dataset_pair"].value_counts()

for pair, count in pair_counts.items():
    pct = count / len(df) * 100
    print(f"{pair:<45} {count:>10,} ({pct:6.2f}%)")

# ------------------------------------------------------------
# Dataset pair × distance
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("3. DATASET PAIR × HAMMING DISTANCE")
print("=" * 70)

pivot = pd.crosstab(
    df["dataset_pair"],
    df["hamming_distance"]
)

print(pivot.to_string())

# ------------------------------------------------------------
# Source-label pairs
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("4. TOP SOURCE-LABEL PAIRS")
print("=" * 70)

df["label_pair"] = df.apply(
    lambda r: " ↔ ".join(
        sorted([
            f"{r['dataset_a']}:{r['source_label_a']}",
            f"{r['dataset_b']}:{r['source_label_b']}",
        ])
    ),
    axis=1,
)

label_counts = df["label_pair"].value_counts().head(30)

for pair, count in label_counts.items():
    print(f"{count:>10,}  {pair}")

# ------------------------------------------------------------
# Hamming 0 only
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("5. HAMMING DISTANCE 0")
print("=" * 70)

h0 = df[df["hamming_distance"] == 0]

print(f"Distance-0 candidate pairs: {len(h0):,}")

if len(h0):
    print("\nDataset-pair distribution for distance 0:")

    h0_pairs = h0["dataset_pair"].value_counts()

    for pair, count in h0_pairs.items():
        print(f"{pair:<45} {count:>10,}")

# ------------------------------------------------------------
# Save summary CSVs
# ------------------------------------------------------------

pair_out = ROOT / "perceptual_candidate_dataset_pairs.csv"
distance_out = ROOT / "perceptual_candidate_distance_summary.csv"
label_out = ROOT / "perceptual_candidate_label_pairs.csv"
h0_out = ROOT / "perceptual_distance0_candidates.csv"

pair_counts.reset_index(
    name="candidate_count"
).to_csv(pair_out, index=False)

distance_counts.reset_index(
    name="candidate_count"
).to_csv(distance_out, index=False)

label_counts.reset_index(
    name="candidate_count"
).to_csv(label_out, index=False)

h0.to_csv(h0_out, index=False)

summary = {
    "total_candidates": int(len(df)),
    "distance_distribution": {
        str(int(k)): int(v)
        for k, v in distance_counts.items()
    },
    "dataset_pairs": {
        str(k): int(v)
        for k, v in pair_counts.items()
    },
    "distance_0_candidates": int(len(h0)),
}

summary_path = ROOT / "perceptual_candidate_analysis_summary.json"

with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 70)
print("ANALYSIS FILES CREATED")
print("=" * 70)

print(pair_out)
print(distance_out)
print(label_out)
print(h0_out)
print(summary_path)

print("\nDO NOT DELETE OR MODIFY ANY SOURCE DATASET.")