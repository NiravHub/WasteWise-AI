from pathlib import Path
from collections import defaultdict
import csv

ROOT = Path("results/dataset_audit/cross_dataset_duplicates")
CSV_PATH = ROOT / "cross_dataset_perceptual_candidates.csv"

print("=" * 70)
print("INSPECTING PERCEPTUAL-HASH COLLISIONS")
print("=" * 70)

# We only need the candidate CSV.
# Build groups based on the actual candidate relationships.

groups = defaultdict(set)

with CSV_PATH.open(
    "r",
    encoding="utf-8",
    newline=""
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        if int(row["hamming_distance"]) != 0:
            continue

        # Each path is represented in a collision group.
        a = row["path_a"]
        b = row["path_b"]

        groups[(row["dataset_a"], row["source_label_a"])].add(b)
        groups[(row["dataset_b"], row["source_label_b"])].add(a)

print()
print(f"Distance-0 candidate relationships: {sum(len(v) for v in groups.values()):,}")

print()
print("=" * 70)
print("TOP SOURCE GROUPS INVOLVED IN DISTANCE-0 CANDIDATES")
print("=" * 70)

group_sizes = sorted(
    (
        (dataset, label, len(paths))
        for (dataset, label), paths in groups.items()
    ),
    key=lambda x: x[2],
    reverse=True
)

for dataset, label, count in group_sizes[:40]:
    print(
        f"{dataset:25s} | "
        f"{label:30s} | "
        f"{count:10,}"
    )

print()
print("=" * 70)
print("IMPORTANT")
print("=" * 70)
print(
    "These are perceptual-hash collision relationships, "
    "NOT confirmed duplicate images."
)