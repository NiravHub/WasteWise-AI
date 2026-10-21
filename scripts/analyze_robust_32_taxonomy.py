import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MANIFEST = (
    ROOT
    / "data"
    / "processed"
    / "taco_detection_complete"
    / "class_manifest.csv"
)

OUTPUT = (
    ROOT
    / "results"
    / "dataset_audit"
    / "taco"
    / "robust_32_taxonomy_analysis.txt"
)


# ============================================================
# 32-CLASS ROBUST TAXONOMY
# ============================================================

TAXONOMY = {
    "cigarette": [
        "Cigarette",
    ],

    "unlabeled_litter": [
        "Unlabeled litter",
    ],

    "plastic_film_wrapper": [
        "Plastic film",
        "Other plastic wrapper",
        "Crisp packet",
    ],

    "plastic_bottle": [
        "Clear plastic bottle",
        "Other plastic bottle",
    ],

    "other_plastic": [
        "Other plastic",
    ],

    "bottle_cap": [
        "Plastic bottle cap",
        "Metal bottle cap",
    ],

    "plastic_straw": [
        "Plastic straw",
        "Paper straw",
    ],

    "glass_bottle": [
        "Glass bottle",
    ],

    "broken_glass": [
        "Broken glass",
    ],

    "glass_container": [
        "Glass jar",
        "Glass cup",
    ],

    "plastic_cup": [
    "Disposable plastic cup",
    "Foam cup",
    "Other plastic cup",
    "Paper cup",
    ],

    "paper": [
        "Normal paper",
        "Magazine paper",
        "Wrapping paper",
        "Tissues",
    ],

    "carton": [
        "Other carton",
        "Drink carton",
        "Corrugated carton",
        "Meal carton",
        "Egg carton",
        "Pizza box",
    ],

    "metal_can": [
        "Drink can",
        "Food Can",
    ],

    "metal_misc": [
        "Aerosol",
        "Scrap metal",
    ],

    "pop_tab": [
        "Pop tab",
    ],

    "plastic_lid": [
        "Plastic lid",
        "Metal lid",
    ],

    "plastic_bag": [
        "Garbage bag",
        "Single-use carrier bag",
        "Polypropylene bag",
    ],

    "plastic_container": [
        "Disposable food container",
        "Foam food container",
        "Other plastic container",
        "Tupperware",
    ],

    "styrofoam": [
        "Styrofoam piece",
    ],

    "plastic_utensil": [
        "Plastic utensils",
    ],

    "rope_string": [
        "Rope & strings",
    ],

    "paper_bag": [
        "Paper bag",
    ],

    "aluminium_foil": [
        "Aluminium foil",
    ],

    "six_pack_ring": [
        "Six pack rings",
    ],

    "medical_packaging": [
        "Aluminium blister pack",
        "Carded blister pack",
    ],

    "food_waste": [
        "Food waste",
    ],

    "shoe": [
        "Shoe",
    ],

    "squeezable_tube": [
        "Squeezable tube",
    ],

    "toilet_tube": [
        "Toilet tube",
    ],

    "spread_tub": [
        "Spread tub",
    ],

    "plastic_gloves": [
        "Plastic glooves",
    ],

    "battery": [
        "Battery",
    ],
}


# ============================================================
# LOAD ORIGINAL TACO CLASS COUNTS
# ============================================================

rows = []

with open(MANIFEST, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        rows.append({
            "taco_id": int(row["taco_id"]),
            "name": row["class_name"],
            "count": int(row["annotation_count"]),
        })


source_counts = {
    row["name"]: row["count"]
    for row in rows
}


# ============================================================
# VALIDATE TAXONOMY
# ============================================================

all_source_classes = set(source_counts)

mapped_classes = set()

for robust_class, original_classes in TAXONOMY.items():
    for original in original_classes:
        mapped_classes.add(original)


missing = sorted(all_source_classes - mapped_classes)
unknown = sorted(mapped_classes - all_source_classes)


# ============================================================
# CALCULATE ROBUST COUNTS
# ============================================================

robust_counts = {}

for robust_class, original_classes in TAXONOMY.items():
    robust_counts[robust_class] = sum(
        source_counts.get(name, 0)
        for name in original_classes
    )


total_source = sum(source_counts.values())
total_robust = sum(robust_counts.values())


# ============================================================
# SORT
# ============================================================

ranked = sorted(
    robust_counts.items(),
    key=lambda x: x[1],
    reverse=True
)


# ============================================================
# IMBALANCE
# ============================================================

positive_counts = [
    count for count in robust_counts.values()
    if count > 0
]

max_count = max(positive_counts)
min_count = min(positive_counts)

imbalance_ratio = max_count / min_count


# ============================================================
# REPORT
# ============================================================

lines = []

lines.append("=" * 80)
lines.append("TACO 32-CLASS ROBUST TAXONOMY VALIDATION")
lines.append("=" * 80)

lines.append("")
lines.append("SOURCE")
lines.append("-" * 80)
lines.append(f"Original classes:       {len(source_counts)}")
lines.append(f"Original annotations:   {total_source}")

lines.append("")
lines.append("PROPOSED TAXONOMY")
lines.append("-" * 80)
lines.append(f"Robust classes:         {len(TAXONOMY)}")

lines.append("")
lines.append("CLASS MAPPING")
lines.append("-" * 80)

for robust_class, original_classes in TAXONOMY.items():

    total = robust_counts[robust_class]

    lines.append(
        f"{robust_class:<25} {total:>5}"
    )

    for original in original_classes:
        lines.append(
            f"    <- {original:<35} "
            f"{source_counts.get(original, 0):>5}"
        )


lines.append("")
lines.append("VALIDATION")
lines.append("-" * 80)

lines.append(
    f"Mapped source classes:   {len(mapped_classes)}"
)

lines.append(
    f"Unmapped source classes:  {len(missing)}"
)

lines.append(
    f"Unknown mapped classes:   {len(unknown)}"
)

lines.append(
    f"Source annotations:       {total_source}"
)

lines.append(
    f"Robust annotations:       {total_robust}"
)

lines.append(
    f"Annotation retention:     "
    f"{(total_robust / total_source) * 100:.2f}%"
)

lines.append(
    f"Max/min imbalance ratio:  {imbalance_ratio:.2f}x"
)


lines.append("")
lines.append("RARE ROBUST CLASSES")
lines.append("-" * 80)

for name, count in sorted(
    robust_counts.items(),
    key=lambda x: x[1]
):
    if count < 50:
        lines.append(
            f"{count:>5} | {name}"
        )


lines.append("")
lines.append("ROBUST CLASSES BY FREQUENCY")
lines.append("-" * 80)

for name, count in ranked:
    lines.append(
        f"{count:>5} | {name}"
    )


lines.append("")
lines.append("FINAL STATUS")
lines.append("-" * 80)

if not missing and not unknown and total_source == total_robust:
    lines.append(
        "PASS — every original TACO annotation is represented."
    )
else:
    lines.append(
        "FAIL — taxonomy mapping requires correction."
    )

if missing:
    lines.append("")
    lines.append("UNMAPPED SOURCE CLASSES:")
    for name in missing:
        lines.append(f"  - {name}")

if unknown:
    lines.append("")
    lines.append("UNKNOWN MAPPED CLASSES:")
    for name in unknown:
        lines.append(f"  - {name}")


OUTPUT.parent.mkdir(parents=True, exist_ok=True)

OUTPUT.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8"
)


# ============================================================
# CONSOLE OUTPUT
# ============================================================

print("=" * 80)
print("TACO 32-CLASS ROBUST TAXONOMY VALIDATION")
print("=" * 80)

print(f"Original classes:       {len(source_counts)}")
print(f"Robust classes:         {len(TAXONOMY)}")
print(f"Original annotations:   {total_source}")
print(f"Robust annotations:     {total_robust}")
print(
    f"Annotation retention:   "
    f"{(total_robust / total_source) * 100:.2f}%"
)
print(
    f"Imbalance ratio:        {imbalance_ratio:.2f}x"
)

print("")
print(f"Unmapped classes:        {len(missing)}")
print(f"Unknown mapped classes:  {len(unknown)}")

print("")
print("Report:")
print(OUTPUT)

print("")

if not missing and not unknown and total_source == total_robust:
    print("PASS — every original TACO annotation is represented.")
else:
    print("FAIL — taxonomy mapping requires correction.")
