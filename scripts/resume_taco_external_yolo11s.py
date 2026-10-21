from ultralytics import YOLO

LAST_CHECKPOINT = (
    r"runs\detect\results\taco_external"
    r"\yolo11s_combined_baseline-2\weights\last.pt"
)

print("=" * 70)
print("RESUMING TACO + EXTERNAL YOLO11s TRAINING")
print("=" * 70)
print(f"Checkpoint: {LAST_CHECKPOINT}")
print("Workers: 0")
print("Device: GPU 0")
print("=" * 70)

model = YOLO(LAST_CHECKPOINT)

model.train(
    resume=True,
    workers=0,
    device=0
)

print("=" * 70)
print("RESUME TRAINING FINISHED")
print("=" * 70)