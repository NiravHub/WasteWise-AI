from pathlib import Path
from collections import defaultdict
from multiprocessing import freeze_support
import yaml

from ultralytics import YOLO


freeze_support()


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT
    / "runs"
    / "detect"
    / "results"
    / "taco_external_balanced"
    / "yolo11s_balanced"
    / "weights"
    / "best.pt"
)

DATASET_ROOT = (
    ROOT
    / "data"
    / "processed"
    / "garbage_external_wastewise"
)

TEST_IMAGES = DATASET_ROOT / "test" / "images"
TEST_LABELS = DATASET_ROOT / "test" / "labels"

TEMP_YAML = DATASET_ROOT / "_evaluation_data.yaml"


# ============================================================
# CLASS DEFINITIONS
# ============================================================

CLASS_NAMES = {
    0: "plastic",
    1: "glass",
    2: "metal",
    3: "paper_cardboard",
    4: "biological",
}


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# CREATE EVALUATION YAML
# ============================================================

def create_evaluation_yaml():

    config = {
        "path": str(DATASET_ROOT.resolve()).replace("\\", "/"),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": CLASS_NAMES,
    }

    with open(TEMP_YAML, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            config,
            f,
            sort_keys=False,
        )

    return TEMP_YAML


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("WASTEWISE YOLO11s EXTERNAL GARBAGE DETECTOR BENCHMARK")
    print("=" * 80)

    print()
    print("Project root:")
    print(f"  {ROOT}")

    print()
    print("Model:")
    print(f"  {MODEL_PATH}")

    print()
    print("Dataset:")
    print(f"  {DATASET_ROOT}")

    # ========================================================
    # PATH VALIDATION
    # ========================================================

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"\nModel not found:\n{MODEL_PATH}"
        )

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"\nDataset directory not found:\n{DATASET_ROOT}"
        )

    if not TEST_IMAGES.exists():
        raise FileNotFoundError(
            f"\nTest images directory not found:\n{TEST_IMAGES}"
        )

    if not TEST_LABELS.exists():
        raise FileNotFoundError(
            f"\nTest labels directory not found:\n{TEST_LABELS}"
        )

    # ========================================================
    # DATASET AUDIT
    # ========================================================

    print()
    print("=" * 80)
    print("DATASET AUDIT")
    print("=" * 80)

    images = sorted(
        path
        for path in TEST_IMAGES.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    labels = sorted(
        path
        for path in TEST_LABELS.rglob("*.txt")
        if path.is_file()
    )

    image_stems = {path.stem for path in images}
    label_stems = {path.stem for path in labels}

    missing_labels = image_stems - label_stems
    orphan_labels = label_stems - image_stems

    ground_truth_boxes = 0
    empty_labels = 0

    class_counts = defaultdict(int)

    for label_file in labels:

        try:
            text = label_file.read_text(
                encoding="utf-8"
            )
        except UnicodeDecodeError:
            text = label_file.read_text(
                encoding="utf-8",
                errors="ignore"
            )

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        if not lines:
            empty_labels += 1
            continue

        for line in lines:

            parts = line.split()

            if len(parts) != 5:
                continue

            try:
                class_id = int(parts[0])
            except ValueError:
                continue

            ground_truth_boxes += 1
            class_counts[class_id] += 1

    print(f"Test images:        {len(images)}")
    print(f"Test labels:        {len(labels)}")
    print(f"Missing labels:     {len(missing_labels)}")
    print(f"Orphan labels:      {len(orphan_labels)}")
    print(f"Empty labels:       {empty_labels}")
    print(f"Ground-truth boxes: {ground_truth_boxes}")

    print()
    print("GROUND-TRUTH CLASS DISTRIBUTION")
    print("-" * 80)

    for class_id, class_name in CLASS_NAMES.items():

        print(
            f"  {class_id}: "
            f"{class_name:<20}"
            f"{class_counts[class_id]:>8}"
        )

    if missing_labels:
        print()
        print("[WARNING] Missing labels detected.")

    if orphan_labels:
        print()
        print("[WARNING] Orphan labels detected.")

    # ========================================================
    # CREATE TEMPORARY DATA YAML
    # ========================================================

    print()
    print("=" * 80)
    print("CREATING EVALUATION CONFIGURATION")
    print("=" * 80)

    evaluation_yaml = create_evaluation_yaml()

    print()
    print(f"[OK] Evaluation YAML:")
    print(f"  {evaluation_yaml}")

    # ========================================================
    # LOAD MODEL
    # ========================================================

    print()
    print("=" * 80)
    print("LOADING YOLO11s E4 MODEL")
    print("=" * 80)

    model = YOLO(str(MODEL_PATH))

    print()
    print("[OK] Model loaded.")

    # ========================================================
    # EXTERNAL DATASET EVALUATION
    # ========================================================

    print()
    print("=" * 80)
    print("RUNNING EXTERNAL GARBAGE DETECTION TEST")
    print("=" * 80)

    print()
    print("Evaluation policy:")
    print("  - Raw YOLO detector")
    print("  - No ResNet classifier")
    print("  - No OOD")
    print("  - No semantic compatibility gate")
    print("  - External test split only")
    print("  - IoU/AP calculated by Ultralytics")
    print("  - Windows DataLoader workers = 0")

    print()

    results = model.val(
        data=str(evaluation_yaml),
        split="test",
        imgsz=640,
        batch=8,
        conf=0.001,
        iou=0.7,
        device=0,
        workers=0,
        verbose=True,
    )

    # ========================================================
    # OVERALL METRICS
    # ========================================================

    print()
    print("=" * 80)
    print("EXTERNAL DETECTOR RESULTS")
    print("=" * 80)

    metrics = results.box

    precision = float(metrics.mp)
    recall = float(metrics.mr)
    map50 = float(metrics.map50)
    map5095 = float(metrics.map)

    print()
    print(f"Precision:       {precision:.4f}")
    print(f"Recall:          {recall:.4f}")
    print(f"mAP@0.50:        {map50:.4f}")
    print(f"mAP@0.50:0.95:   {map5095:.4f}")

    # ========================================================
    # PER-CLASS METRICS
    # ========================================================

    print()
    print("=" * 80)
    print("PER-CLASS RESULTS")
    print("=" * 80)

    print()

    try:

        class_precision = metrics.p
        class_recall = metrics.r
        class_map50 = metrics.ap50
        class_map = metrics.ap

        print(
            f"{'Class':<22}"
            f"{'GT Boxes':>12}"
            f"{'Precision':>12}"
            f"{'Recall':>12}"
            f"{'mAP50':>12}"
            f"{'mAP50-95':>14}"
        )

        print("-" * 84)

        active_classes = [
            class_id
            for class_id in CLASS_NAMES
            if class_counts[class_id] > 0
        ]

        if len(class_precision) == len(active_classes):

            for metric_index, class_id in enumerate(active_classes):

                class_name = CLASS_NAMES[class_id]

                p = float(class_precision[metric_index])
                r = float(class_recall[metric_index])
                ap50 = float(class_map50[metric_index])
                ap = float(class_map[metric_index])

                print(
                    f"{class_name:<22}"
                    f"{class_counts[class_id]:>12}"
                    f"{p:>12.4f}"
                    f"{r:>12.4f}"
                    f"{ap50:>12.4f}"
                    f"{ap:>14.4f}"
                )

        else:

            print()
            print(
                "[WARNING] Metric-array length does not match "
                "active ground-truth classes."
            )

            print(f"  Active classes: {active_classes}")
            print(f"  Metric entries: {len(class_precision)}")

    except Exception as exc:

        print()
        print(
            "[WARNING] Could not read per-class metrics:"
        )
        print(f"  {exc}")

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)

    print()
    print(f"Test images:            {len(images)}")
    print(f"Ground-truth boxes:     {ground_truth_boxes}")
    print(f"Precision:              {precision:.4f}")
    print(f"Recall:                 {recall:.4f}")
    print(f"mAP@0.50:               {map50:.4f}")
    print(f"mAP@0.50:0.95:          {map5095:.4f}")

    print()
    print("[PASS] External detector benchmark completed.")

    print("=" * 80)


# ============================================================
# WINDOWS ENTRY POINT
# ============================================================

if __name__ == "__main__":
    freeze_support()
    main()