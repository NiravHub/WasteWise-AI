from pathlib import Path
import shutil
import sys
import tempfile

import yaml
from ultralytics import YOLO


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

DATA_ROOT = (
    ROOT
    / "data"
    / "external"
    / "trashnet_plus_plus"
    / "TrashNet"
    / "YOLO Formet"
)

SOURCE_TEST = DATA_ROOT / "test"

# TrashNet++ -> WasteWise detector taxonomy
#
# 0 Cable         -> 0 plastic
# 1 E-waste       -> 5 other_trash
# 2 Glass         -> 1 glass
# 3 Medical Waste -> 5 other_trash
# 4 Metal         -> 2 metal
# 5 Plastics      -> 0 plastic
# 6 can           -> 2 metal

SOURCE_TO_WASTEWISE = {
    0: 0,
    1: 5,
    2: 1,
    3: 5,
    4: 2,
    5: 0,
    6: 2,
}

WASTEWISE_NAMES = {
    0: "plastic",
    1: "glass",
    2: "metal",
    3: "paper_cardboard",
    4: "biological",
    5: "other_trash",
}


def remap_label_file(source_file, destination_file):
    lines = source_file.read_text(encoding="utf-8").splitlines()
    output = []

    for line in lines:
        parts = line.split()

        if len(parts) != 5:
            continue

        source_class = int(parts[0])

        if source_class not in SOURCE_TO_WASTEWISE:
            raise ValueError(
                f"Unknown TrashNet++ class {source_class} "
                f"in {source_file}"
            )

        parts[0] = str(SOURCE_TO_WASTEWISE[source_class])
        output.append(" ".join(parts))

    destination_file.write_text(
        "\n".join(output) + ("\n" if output else ""),
        encoding="utf-8",
    )


def build_evaluation_copy(temp_root):
    temp_test = temp_root / "test"
    temp_images = temp_test / "images"
    temp_labels = temp_test / "labels"

    temp_images.mkdir(parents=True)
    temp_labels.mkdir(parents=True)

    source_images = SOURCE_TEST / "images"
    source_labels = SOURCE_TEST / "labels"

    labeled_count = 0
    unlabeled_count = 0
    box_count = 0

    for image in source_images.iterdir():

        if not image.is_file():
            continue

        label = source_labels / f"{image.stem}.txt"

        # Only evaluate genuinely paired images.
        if not label.exists():
            unlabeled_count += 1
            continue

        shutil.copy2(image, temp_images / image.name)

        destination_label = temp_labels / label.name
        remap_label_file(label, destination_label)

        labeled_count += 1

        for line in label.read_text(encoding="utf-8").splitlines():
            if len(line.split()) == 5:
                box_count += 1

    return (
        temp_test,
        labeled_count,
        unlabeled_count,
        box_count,
    )


def create_yaml(temp_root):
    config = {
        "path": str(temp_root).replace("\\", "/"),
        "train": "test/images",
        "val": "test/images",
        "test": "test/images",
        "names": WASTEWISE_NAMES,
    }

    yaml_path = temp_root / "data.yaml"

    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    return yaml_path


def main():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    print("=" * 70)
    print("WASTEWISE AI - TRASHNET++ EXTERNAL DETECTOR EVALUATION")
    print("=" * 70)

    print(f"Model: {MODEL_PATH}")
    print(f"Source dataset: {DATA_ROOT}")
    print("Split: test")
    print("Evaluation: RAW YOLO DETECTOR")
    print()

    with tempfile.TemporaryDirectory(
        prefix="wastewise_trashnet_eval_"
    ) as temp_dir:

        temp_root = Path(temp_dir)

        (
            temp_test,
            labeled_count,
            unlabeled_count,
            box_count,
        ) = build_evaluation_copy(temp_root)

        data_yaml = create_yaml(temp_root)

        print("Temporary evaluation copy created.")
        print(f"Location: {temp_root}")
        print()
        print("Evaluation dataset:")
        print(f"  Labeled images:   {labeled_count}")
        print(f"  Unlabeled images: {unlabeled_count}")
        print(f"  Ground-truth boxes: {box_count}")
        print()

        model = YOLO(str(MODEL_PATH))

        results = model.val(
            data=str(data_yaml),
            split="test",
            imgsz=640,
            batch=8,
            conf=0.001,
            iou=0.7,
            device=0,
            workers=0,
            verbose=True,
        )

        print()
        print("=" * 70)
        print("FINAL TRASHNET++ RESULTS")
        print("=" * 70)

        print(f"Precision:     {results.box.mp:.4f}")
        print(f"Recall:        {results.box.mr:.4f}")
        print(f"mAP50:         {results.box.map50:.4f}")
        print(f"mAP50-95:      {results.box.map:.4f}")

        print()
        print("Per-class metrics:")

        for i, name in results.names.items():

            if i >= len(results.box.p):
                print(
                    f"{i:2d} {name:18s} "
                    "N/A (no ground-truth instances)"
                )
                continue

            print(
                f"{i:2d} {name:18s} "
                f"P={results.box.p[i]:.4f} "
                f"R={results.box.r[i]:.4f} "
                f"mAP50={results.box.ap50[i]:.4f} "
                f"mAP50-95={results.box.ap[i]:.4f}"
            )

        print()
        print("TrashNet++ test accounting:")
        print(f"  Total images:    {labeled_count + unlabeled_count}")
        print(f"  Labeled images:  {labeled_count}")
        print(f"  Unlabeled images: {unlabeled_count}")
        print(f"  GT boxes:        {box_count}")

        print()
        print("Original source dataset was NOT modified.")
        print("Temporary evaluation copy was automatically deleted.")


if __name__ == "__main__":
    main()