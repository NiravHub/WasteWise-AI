from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "results"
    / "taco_external_balanced"
 / "yolo11s_balanced"
    / "weights"
    / "best.pt"
)

DATA_YAML = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "taco_external_combined"
    / "data.yaml"
)

TACO_TEST_YAML = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "taco_detection_practical"
    / "data.yaml"
)

print("=" * 75)
print("E4 FINAL EVALUATION")
print("TACO TRAIN + EXTERNAL TRAIN -> YOLO11s + CLASS BALANCING")
print("=" * 75)
print(f"Model:       {MODEL_PATH}")
print(f"Test dataset: TACO practical TEST")
print(f"Test config:  {TACO_TEST_YAML}")
print("=" * 75)

model = YOLO(str(MODEL_PATH))

results = model.val(
    data=str(TACO_TEST_YAML),
    split="test",
    imgsz=640,
    batch=16,
    device=0,
    workers=0,
    plots=True,
    project=str(PROJECT_ROOT / "runs" / "detect" / "results" / "taco_external_balanced"),
name="yolo11s_balanced_taco_test",
    exist_ok=True,
)

print("\n" + "=" * 75)
print("E4 FINAL TACO TEST RESULTS")
print("=" * 75)

print(f"Precision:  {results.box.mp:.4f}")
print(f"Recall:     {results.box.mr:.4f}")
print(f"mAP50:      {results.box.map50:.4f}")
print(f"mAP50-95:   {results.box.map:.4f}")

print("=" * 75)
print("PER-CLASS RESULTS")
print("=" * 75)

class_names = [
    "plastic",
    "glass",
    "metal",
    "paper_cardboard",
    "biological",
    "other_trash",
]

for i, name in enumerate(class_names):
    if i < len(results.box.ap50):
        print(
            f"{i}: {name:18s} "
            f"AP50={results.box.ap50[i]:.4f} "
            f"AP50-95={results.box.ap[i]:.4f}"
        )

print("=" * 75)
print("E4 EVALUATION COMPLETE")
print("=" * 75)