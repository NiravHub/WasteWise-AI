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

RUNS_DIR = (
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "results"
    / "taco_external"
)


def main():
    print("=" * 70)
    print("WasteWise AI - E3 TACO + External YOLO11s")
    print("=" * 70)

    print("\nDataset:")
    print(DATA_YAML)

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"data.yaml not found: {DATA_YAML}"
        )

    RUNS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nLoading YOLO11s pretrained model...")

    model = YOLO("yolo11s.pt")

    print("\nStarting training...")

    results = model.train(
        data=str(DATA_YAML),

        epochs=90,
        imgsz=640,
        batch=16,

        device=0,
        workers=4,

        pretrained=True,

        patience=20,

        project=str(RUNS_DIR),
        name="yolo11s_combined_baseline",

        exist_ok=False,

        verbose=True,
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print("\nBest checkpoint should be:")
    print(
        RUNS_DIR
        / "yolo11s_combined_baseline"
        / "weights"
        / "best.pt"
    )


if __name__ == "__main__":
    main()