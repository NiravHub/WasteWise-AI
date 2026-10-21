from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parents[1]

OUT_DIR = ROOT / "results" / "dataset_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUT_DIR / "cross_dataset_taxonomy_audit.csv"
OUTPUT_JSON = OUT_DIR / "cross_dataset_taxonomy_audit.json"


# ============================================================
# SOURCE LABELS
# ============================================================

sources = {
    "Garbage Dataset": [
        "battery",
        "biological",
        "cardboard",
        "clothes",
        "glass",
        "metal",
        "paper",
        "plastic",
        "shoes",
        "trash",
    ],

    "TrashNet++": [
        "Cable",
        "Can",
        "E-wast",
        "Glass",
        "Medical Wast",
        "Metal",
        "Plastic",
    ],

    "Medical Waste 4.0": [
        "gauze",
        "gauzes",
        "glove_pair_latex",
        "glove_pair_nitrile",
        "glove_pair_surgery",
        "glove_single_latex",
        "glove_single_nitrile",
        "glove_single_surgery",
        "medical_cap",
        "medical_filter",
        "medical_glasses",
        "shoe_cover_pair",
        "shoe_cover_single",
        "test_tube",
        "urine_bag",
    ],

    "TACO": [
        "Aerosol",
        "Aluminium blister pack",
        "Aluminium foil",
        "Battery",
        "Broken glass",
        "Carded blister pack",
        "Cigarette",
        "Clear plastic bottle",
        "Corrugated carton",
        "Crisp packet",
        "Disposable food container",
        "Disposable plastic cup",
        "Drink can",
        "Drink carton",
        "Egg carton",
        "Foam cup",
        "Foam food container",
        "Food Can",
        "Food waste",
        "Garbage bag",
        "Glass bottle",
        "Glass cup",
        "Glass jar",
        "Magazine paper",
        "Meal carton",
        "Metal bottle cap",
        "Metal lid",
        "Normal paper",
        "Other carton",
        "Other plastic",
        "Other plastic bottle",
        "Other plastic container",
        "Other plastic cup",
        "Other plastic wrapper",
        "Paper bag",
        "Paper cup",
        "Paper straw",
        "Pizza box",
        "Plastic bottle cap",
        "Plastic film",
        "Plastic glooves",
        "Plastic lid",
        "Plastic straw",
        "Plastic utensils",
        "Plastified paper bag",
        "Polypropylene bag",
        "Pop tab",
        "Rope & strings",
        "Scrap metal",
        "Shoe",
        "Single-use carrier bag",
        "Six pack rings",
        "Spread tub",
        "Squeezable tube",
        "Styrofoam piece",
        "Tissues",
        "Toilet tube",
        "Tupperware",
        "Unlabeled litter",
        "Wrapping paper",
    ],
}


# ============================================================
# CONSERVATIVE TAXONOMY
# ============================================================
#
# Important:
# This is a PROPOSED taxonomy for audit purposes.
# It does NOT modify or merge any dataset.
#
# "review_required" means we should inspect examples before
# allowing the label into the final training taxonomy.
#

MAPPING = {}


def add(source, labels, family, canonical, action, notes=""):
    for label in labels:
        MAPPING[(source, label)] = {
            "waste_family": family,
            "canonical_class": canonical,
            "recommended_action": action,
            "notes": notes,
        }


# ---------------- Garbage Dataset ----------------

add(
    "Garbage Dataset",
    ["battery"],
    "e_waste",
    "battery",
    "classification_core",
    "Battery is both e-waste and potentially hazardous; keep hazard metadata separately.",
)

add(
    "Garbage Dataset",
    ["biological"],
    "organic_biological",
    "biological_waste",
    "classification_core",
    "Do not assume medical contamination from this label.",
)

add(
    "Garbage Dataset",
    ["cardboard"],
    "recyclable",
    "cardboard",
    "classification_core",
)

add(
    "Garbage Dataset",
    ["clothes", "shoes"],
    "textile_reusable",
    "textile",
    "classification_core",
    "Clothing/shoes may be reusable rather than waste; preserve this semantic distinction in recommendations.",
)

add(
    "Garbage Dataset",
    ["glass"],
    "recyclable",
    "glass",
    "classification_core",
)

add(
    "Garbage Dataset",
    ["metal"],
    "recyclable",
    "metal",
    "classification_core",
)

add(
    "Garbage Dataset",
    ["paper"],
    "recyclable",
    "paper",
    "classification_core",
)

add(
    "Garbage Dataset",
    ["plastic"],
    "recyclable",
    "plastic",
    "classification_core",
)

add(
    "Garbage Dataset",
    ["trash"],
    "residual",
    "general_trash",
    "classification_core",
)


# ---------------- TrashNet++ ----------------

add(
    "TrashNet++",
    ["Cable"],
    "e_waste",
    "cable",
    "classification_expansion",
)

add(
    "TrashNet++",
    ["Can"],
    "recyclable",
    "metal_can",
    "review_required",
    "Confirm whether all source Can examples are metal cans before merging.",
)

add(
    "TrashNet++",
    ["E-wast"],
    "e_waste",
    "electronic_waste",
    "classification_expansion",
)

add(
    "TrashNet++",
    ["Glass"],
    "recyclable",
    "glass",
    "classification_expansion",
)

add(
    "TrashNet++",
    ["Medical Wast"],
    "medical",
    "medical_waste_object",
    "review_required",
    "Visual object classification does not establish contamination or infection risk.",
)

add(
    "TrashNet++",
    ["Metal"],
    "recyclable",
    "metal",
    "classification_expansion",
)

add(
    "TrashNet++",
    ["Plastic"],
    "recyclable",
    "plastic",
    "classification_expansion",
)


# ---------------- Medical Waste 4.0 ----------------

add(
    "Medical Waste 4.0",
    ["gauze", "gauzes"],
    "medical",
    "gauze",
    "classification_medical",
    "gauze/gauzes are retained as source labels for now; canonical merge can be decided after image review.",
)

add(
    "Medical Waste 4.0",
    [
        "glove_pair_latex",
        "glove_pair_nitrile",
        "glove_pair_surgery",
        "glove_single_latex",
        "glove_single_nitrile",
        "glove_single_surgery",
    ],
    "medical",
    "medical_gloves",
    "classification_medical",
    "Pair/single and material information can be retained as metadata.",
)

add(
    "Medical Waste 4.0",
    ["medical_cap"],
    "medical",
    "medical_cap",
    "classification_medical",
)

add(
    "Medical Waste 4.0",
    ["medical_filter"],
    "medical",
    "medical_filter",
    "classification_medical",
)

add(
    "Medical Waste 4.0",
    ["medical_glasses"],
    "medical",
    "medical_glasses",
    "classification_medical",
)

add(
    "Medical Waste 4.0",
    ["shoe_cover_pair", "shoe_cover_single"],
    "medical",
    "medical_shoe_cover",
    "classification_medical",
)

add(
    "Medical Waste 4.0",
    ["test_tube"],
    "medical",
    "test_tube",
    "classification_medical",
)

add(
    "Medical Waste 4.0",
    ["urine_bag"],
    "medical",
    "urine_bag",
    "classification_medical",
)


# ---------------- TACO ----------------
#
# TACO is primarily useful for real-world litter / detection.
# We therefore DO NOT automatically merge its fine-grained
# labels into the current image-classification taxonomy.
#

taco_recyclable = [
    "Aluminium blister pack",
    "Aluminium foil",
    "Clear plastic bottle",
    "Corrugated carton",
    "Drink can",
    "Drink carton",
    "Egg carton",
    "Glass bottle",
    "Glass cup",
    "Glass jar",
    "Magazine paper",
    "Meal carton",
    "Metal bottle cap",
    "Metal lid",
    "Normal paper",
    "Other carton",
    "Other plastic",
    "Other plastic bottle",
    "Other plastic container",
    "Other plastic cup",
    "Other plastic wrapper",
    "Paper bag",
    "Paper cup",
    "Paper straw",
    "Pizza box",
    "Plastic bottle cap",
    "Plastic film",
    "Plastic lid",
    "Plastic straw",
    "Plastic utensils",
    "Plastified paper bag",
    "Polypropylene bag",
    "Pop tab",
    "Scrap metal",
    "Single-use carrier bag",
    "Six pack rings",
    "Spread tub",
    "Tupperware",
    "Wrapping paper",
]

add(
    "TACO",
    taco_recyclable,
    "recyclable",
    "taco_packaging_object",
    "detection_candidate",
    "Keep TACO object labels separate from the current classification taxonomy.",
)

add(
    "TACO",
    ["Battery"],
    "e_waste",
    "battery",
    "detection_candidate",
    "Useful for future object detection; do not automatically merge with classification images.",
)

add(
    "TACO",
    ["Aerosol"],
    "hazardous_special",
    "aerosol_container",
    "detection_candidate",
    "Image alone cannot establish chemical contents/toxicity.",
)

add(
    "TACO",
    ["Broken glass"],
    "hazardous_special",
    "broken_glass",
    "detection_candidate",
)

add(
    "TACO",
    ["Cigarette"],
    "hazardous_special",
    "cigarette",
    "detection_candidate",
)

add(
    "TACO",
    ["Food waste"],
    "organic_biological",
    "food_waste",
    "detection_candidate",
)

add(
    "TACO",
    ["Shoe"],
    "textile_reusable",
    "shoe",
    "detection_candidate",
)

add(
    "TACO",
    ["Tissues"],
    "residual",
    "tissue",
    "detection_candidate",
)

add(
    "TACO",
    ["Unlabeled litter"],
    "residual",
    "unlabeled_litter",
    "detection_candidate",
)

add(
    "TACO",
    [
        "Carded blister pack",
        "Crisp packet",
        "Disposable food container",
        "Disposable plastic cup",
        "Foam cup",
        "Foam food container",
        "Garbage bag",
        "Other plastic",
        "Other plastic wrapper",
        "Rope & strings",
        "Squeezable tube",
        "Styrofoam piece",
        "Toilet tube",
    ],
    "mixed_packaging",
    "packaging_object",
    "review_required",
    "Object-level labels require image review before deciding canonical classification family.",
)


# ============================================================
# GENERATE AUDIT
# ============================================================

rows = []

for source, labels in sources.items():
    for label in labels:

        mapping = MAPPING.get(
            (source, label),
            {
                "waste_family": "unmapped",
                "canonical_class": "unmapped",
                "recommended_action": "review_required",
                "notes": "No preliminary mapping was assigned.",
            },
        )

        rows.append(
            {
                "source": source,
                "original_label": label,
                **mapping,
            }
        )


# ============================================================
# WRITE CSV
# ============================================================

fieldnames = [
    "source",
    "original_label",
    "waste_family",
    "canonical_class",
    "recommended_action",
    "notes",
]

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


# ============================================================
# SUMMARY
# ============================================================

summary = {
    "sources": {},
    "actions": {},
    "families": {},
}

for row in rows:

    source = row["source"]
    action = row["recommended_action"]
    family = row["waste_family"]

    summary["sources"][source] = summary["sources"].get(source, 0) + 1
    summary["actions"][action] = summary["actions"].get(action, 0) + 1
    summary["families"][family] = summary["families"].get(family, 0) + 1


output = {
    "purpose": "Cross-dataset taxonomy audit before preprocessing and merging.",
    "warning": (
        "This mapping is preliminary. It does not modify datasets and "
        "does not authorize automatic merging."
    ),
    "rows": rows,
    "summary": summary,
}

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)


# ============================================================
# CONSOLE REPORT
# ============================================================

print("=" * 70)
print("WASTEWISE AI - PHASE 3 TAXONOMY AUDIT")
print("=" * 70)

print("\nSOURCE LABEL COUNTS")
for source, count in summary["sources"].items():
    print(f"  {source}: {count}")

print("\nRECOMMENDED ACTIONS")
for action, count in summary["actions"].items():
    print(f"  {action}: {count}")

print("\nWASTE FAMILIES")
for family, count in summary["families"].items():
    print(f"  {family}: {count}")

print("\nOUTPUT FILES")
print(f"  CSV : {OUTPUT_CSV}")
print(f"  JSON: {OUTPUT_JSON}")

print("\nIMPORTANT")
print("  No dataset files were modified.")
print("  No images were copied, resized, merged, or deleted.")
print("  COCO remains an OOD/non-waste resource.")
print("  TACO remains primarily a detection/litter resource.")
print("=" * 70)