from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_YAML = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "taco_external_combined"
    / "data.yaml"
)

MODEL = "yolo11s.pt"

print("=" * 80)
print("E4 — BALANCED TACO + EXTERNAL YOLO11s")
print("=" * 80)
print(f"Dataset:     {DATA_YAML}")
print(f"Model:       {MODEL}")
print("Epochs:      90")
print("Image size:  640")
print("Batch:       16")
print("Workers:     0")
print("Device:      GPU 0")
print("Class power: cls_pw=0.5")
print("=" * 80)

model = YOLO(MODEL)

model.train(
    data=str(DATA_YAML),
    epochs=90,
    imgsz=640,
    batch=16,
    device=0,
    workers=0,
    patience=20,

    # Class-balancing:
    # inverse-frequency weights raised to the power 0.5
    cls_pw=0.5,

    project=str(
        PROJECT_ROOT
        / "runs"
        / "detect"
        / "results"
        / "taco_external_balanced"
    ),
    name="yolo11s_balanced",
)

print("=" * 80)
print("E4 TRAINING COMPLETE")
print("=" * 80)