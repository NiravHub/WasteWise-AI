from pathlib import Path
import shutil
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "data" / "processed" / "garbage_external_clean"
OUTPUT = ROOT / "data" / "processed" / "garbage_external_wastewise"

SPLITS = ["train", "valid", "test"]

# External dataset:
# 0 BIODEGRADABLE
# 1 CARDBOARD
# 2 GLASS
# 3 METAL
# 4 PAPER
# 5 PLASTIC

SOURCE_NAMES = [
    "BIODEGRADABLE",
    "CARDBOARD",
    "GLASS",
    "METAL",
    "PAPER",
    "PLASTIC",
]

# WasteWise practical detector taxonomy.
TARGET_NAMES = [
    "plastic",
    "glass",
    "metal",
    "paper_cardboard",
    "biological",
]

CLASS_MAP = {
    0: 4,  # BIODEGRADABLE -> biological
    1: 3,  # CARDBOARD -> paper_cardboard
    2: 1,  # GLASS -> glass
    3: 2,  # METAL -> metal
    4: 3,  # PAPER -> paper_cardboard
    5: 0,  # PLASTIC -> plastic
}

for split in SPLITS:
    (OUTPUT / split / "images").mkdir(
        parents=True,
        exist_ok=True
    )
    (OUTPUT / split / "labels").mkdir(
        parents=True,
        exist_ok=True
    )

counts = Counter()
total_boxes = 0
mapped_boxes = 0

for split in SPLITS:

    src_images = SOURCE / split / "images"
    src_labels = SOURCE / split / "labels"

    dst_images = OUTPUT / split / "images"
    dst_labels = OUTPUT / split / "labels"

    for image_path in src_images.iterdir():

        if not image_path.is_file():
            continue

        shutil.copy2(
            image_path,
            dst_images / image_path.name
        )

    for label_path in src_labels.glob("*.txt"):

        output_lines = []

        with open(label_path, "r", encoding="utf-8") as f:

            for line in f:
                line = line.strip()

                if not line:
                    continue

                parts = line.split()

                if len(parts) != 5:
                    raise ValueError(
                        f"Malformed label: "
                        f"{split}/{label_path.name}: {line}"
                    )

                source_class = int(float(parts[0]))

                if source_class not in CLASS_MAP:
                    raise ValueError(
                        f"Unknown class {source_class} "
                        f"in {label_path}"
                    )

                target_class = CLASS_MAP[source_class]

                output_lines.append(
                    " ".join(
                        [
                            str(target_class),
                            *parts[1:]
                        ]
                    )
                )

                counts[TARGET_NAMES[target_class]] += 1
                total_boxes += 1
                mapped_boxes += 1

        output_label = dst_labels / label_path.name

        with open(output_label, "w", encoding="utf-8") as f:
            if output_lines:
                f.write("\n".join(output_lines) + "\n")

print("=" * 70)
print("WASTEWISE EXTERNAL TAXONOMY MAPPING COMPLETE")
print("=" * 70)

print("\nSOURCE CLASSES")
for i, name in enumerate(SOURCE_NAMES):
    print(f"{i}: {name}")

print("\nTARGET CLASSES")
for i, name in enumerate(TARGET_NAMES):
    print(f"{i}: {name}")

print("\nMAPPING")
for source_id, target_id in CLASS_MAP.items():
    print(
        f"{SOURCE_NAMES[source_id]:<15} -> "
        f"{TARGET_NAMES[target_id]}"
    )

print("\nCLASS DISTRIBUTION")

for name in TARGET_NAMES:
    print(
        f"{name:<20} {counts[name]:>8}"
    )

print("\nTOTAL BOXES:", total_boxes)
print("MAPPED BOXES:", mapped_boxes)

print("\nOUTPUT:")
print(OUTPUT)

print("\nPASS — external dataset mapped to WasteWise taxonomy.")