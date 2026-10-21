import csv
from collections import Counter, defaultdict
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
    / "robust_taxonomy_analysis.txt"
)

rows = []

with open(MANIFEST, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append({
            "yolo_id": int(row["yolo_id"]),
            "taco_id": int(row["taco_id"]),
            "name": row["class_name"],
            "count": int(row["annotation_count"]),
        })


# ------------------------------------------------------------
# Sort by frequency
# ------------------------------------------------------------

by_frequency = sorted(
    rows,
    key=lambda x: x["count"],
    reverse=True
)


rare_10 = [r for r in rows if r["count"] < 10]
rare_20 = [r for r in rows if r["count"] < 20]
rare_50 = [r for r in rows if r["count"] < 50]


# ------------------------------------------------------------
# Candidate semantic groups
#
# IMPORTANT:
# These are ONLY candidates.
# They are NOT automatically applied.
# ------------------------------------------------------------

groups = {
    "plastic_bottles": [
        "Clear plastic bottle",
        "Other plastic bottle",
    ],

    "bottle_caps": [
        "Plastic bottle cap",
        "Metal bottle cap",
    ],

    "cartons": [
        "Other carton",
        "Drink carton",
        "Corrugated carton",
        "Meal carton",
        "Egg carton",
        "Pizza box",
    ],

    "plastic_cups": [
        "Disposable plastic cup",
        "Foam cup",
        "Glass cup",
        "Other plastic cup",
    ],

    "paper_products": [
        "Normal paper",
        "Magazine paper",
        "Wrapping paper",
        "Tissues",
    ],

    "paper_bags": [
        "Paper bag",
        "Plastified paper bag",
    ],

    "plastic_film_wrappers": [
        "Plastic film",
        "Other plastic wrapper",
        "Crisp packet",
    ],

    "plastic_bags": [
        "Garbage bag",
        "Single-use carrier bag",
        "Polypropylene bag",
    ],

    "plastic_containers": [
        "Disposable food container",
        "Foam food container",
        "Other plastic container",
        "Tupperware",
    ],

    "plastic_lids": [
        "Plastic lid",
        "Metal lid",
    ],

    "straws": [
        "Plastic straw",
        "Paper straw",
    ],

    "foam_items": [
        "Styrofoam piece",
        "Foam food container",
        "Foam cup",
    ],

    "metal_items": [
        "Drink can",
        "Food Can",
        "Aerosol",
        "Scrap metal",
        "Pop tab",
    ],

    "glass_items": [
        "Glass bottle",
        "Broken glass",
        "Glass jar",
        "Glass cup",
    ],

    "plastic_misc": [
        "Other plastic",
        "Plastic utensils",
        "Plastic lid",
        "Plastic bottle cap",
    ],
}


name_to_count = {
    r["name"]: r["count"]
    for r in rows
}


# ------------------------------------------------------------
# Write report
# ------------------------------------------------------------

lines = []

lines.append("=" * 80)
lines.append("TACO ROBUST TAXONOMY ANALYSIS")
lines.append("=" * 80)

lines.append("")
lines.append("SOURCE")
lines.append("-" * 80)
lines.append(f"Classes:       {len(rows)}")
lines.append(
    f"Annotations:   {sum(r['count'] for r in rows)}"
)

lines.append("")
lines.append("ALL CLASSES BY FREQUENCY")
lines.append("-" * 80)

for r in by_frequency:
    lines.append(
        f"{r['count']:4d} | "
        f"TACO {r['taco_id']:2d} | "
        f"{r['name']}"
    )

lines.append("")
lines.append("RARE CLASS SUMMARY")
lines.append("-" * 80)
lines.append(
    f"< 10 annotations: {len(rare_10)} classes"
)
lines.append(
    f"< 20 annotations: {len(rare_20)} classes"
)
lines.append(
    f"< 50 annotations: {len(rare_50)} classes"
)

lines.append("")
lines.append("CLASSES WITH <10 ANNOTATIONS")
lines.append("-" * 80)

for r in sorted(rare_10, key=lambda x: x["count"]):
    lines.append(
        f"{r['count']:4d} | {r['name']}"
    )


lines.append("")
lines.append("CLASSES WITH <20 ANNOTATIONS")
lines.append("-" * 80)

for r in sorted(rare_20, key=lambda x: x["count"]):
    lines.append(
        f"{r['count']:4d} | {r['name']}"
    )


lines.append("")
lines.append("CLASSES WITH <50 ANNOTATIONS")
lines.append("-" * 80)

for r in sorted(rare_50, key=lambda x: x["count"]):
    lines.append(
        f"{r['count']:4d} | {r['name']}"
    )


lines.append("")
lines.append("CANDIDATE SEMANTIC GROUPS")
lines.append("-" * 80)
lines.append(
    "These groups are candidates for review only."
)
lines.append(
    "They must NOT be applied automatically."
)

for group_name, class_list in groups.items():

    available = [
        name for name in class_list
        if name in name_to_count
    ]

    total = sum(
        name_to_count[name]
        for name in available
    )

    lines.append("")
    lines.append(
        f"[{group_name}] total={total}"
    )

    for name in available:
        lines.append(
            f"    {name}: {name_to_count[name]}"
        )


OUTPUT.parent.mkdir(parents=True, exist_ok=True)

OUTPUT.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8"
)


# ------------------------------------------------------------
# Console summary
# ------------------------------------------------------------

print("=" * 80)
print("TACO ROBUST TAXONOMY ANALYSIS COMPLETE")
print("=" * 80)

print(f"Classes:              {len(rows)}")
print(f"Annotations:          {sum(r['count'] for r in rows)}")
print(f"<10 annotations:      {len(rare_10)} classes")
print(f"<20 annotations:      {len(rare_20)} classes")
print(f"<50 annotations:      {len(rare_50)} classes")

print("")
print("Report:")
print(OUTPUT)

print("")
print("IMPORTANT:")
print("Candidate groups are NOT automatically applied.")
print("=" * 80)