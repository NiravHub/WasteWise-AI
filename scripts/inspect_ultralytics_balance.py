import inspect
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

print("=" * 80)
print("ULTRALYTICS VERSION / BALANCING INSPECTION")
print("=" * 80)

import ultralytics
print("Ultralytics version:", ultralytics.__version__)

print("\nDetectionTrainer:")
print(inspect.getsource(DetectionTrainer))

print("\nSearching trainer source for balancing-related parameters...")
source = inspect.getsource(DetectionTrainer)

keywords = [
    "cls_pw",
    "class_weight",
    "class_weights",
    "weight",
    "loss",
]

for keyword in keywords:
    print(f"{keyword}: {keyword in source}")

print("=" * 80)