from pathlib import Path
from collections import Counter
import sys

# ============================================================================
# MAKE PROJECT ROOT IMPORTABLE
# ============================================================================

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import torch
from PIL import Image
from torchvision import transforms

from src.models.resnet50_model import WasteResNet50


# ============================================================================
# PROJECT PATHS
# ============================================================================

MODEL_PATH = (
    ROOT
    / "models"
    / "expanded_resnet50_stage2_best.pth"
)

DATASET_PATH = (
    ROOT
    / "data"
    / "raw"
    / "garbage-dataset"
)


# ============================================================================
# SETTINGS
# ============================================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


# ============================================================================
# ORIGINAL DATASET -> WASTEWISE CLASS MAPPING
# ============================================================================

LABEL_MAP = {
    "battery": "battery",
    "biological": "biological",
    "cardboard": "cardboard",
    "clothes": "clothes",
    "glass": "glass",
    "metal": "metal",
    "paper": "paper",
    "plastic": "plastic",
    "shoes": "shoes",
    "trash": "general_trash",
}


# ============================================================================
# HEADER
# ============================================================================

print("=" * 80)
print("WASTEWISE RESNET50 FULL CLASSIFIER BENCHMARK")
print("=" * 80)

print()
print("Project root:")
print(f"  {ROOT}")

print()
print(f"Device: {DEVICE}")

if DEVICE.type == "cuda":
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

print()
print("Model:")
print(f"  {MODEL_PATH}")

print()
print("Dataset:")
print(f"  {DATASET_PATH}")


# ============================================================================
# VALIDATE PATHS
# ============================================================================

if not MODEL_PATH.exists():
    print()
    print("ERROR: Classifier model not found:")
    print(MODEL_PATH)
    sys.exit(1)


if not DATASET_PATH.exists():
    print()
    print("ERROR: Dataset not found:")
    print(DATASET_PATH)
    sys.exit(1)


# ============================================================================
# EXACT PRODUCTION PREPROCESSING
# ============================================================================

RESNET_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


# ============================================================================
# LOAD CHECKPOINT
# ============================================================================

print()
print("=" * 80)
print("LOADING CHECKPOINT")
print("=" * 80)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
)


if not isinstance(checkpoint, dict):
    print()
    print("ERROR: Unexpected checkpoint format.")
    print(f"Checkpoint type: {type(checkpoint)}")
    sys.exit(1)


print()
print("Checkpoint keys:")

for key in checkpoint.keys():
    print(f"  - {key}")


# ============================================================================
# CHECKPOINT METADATA
# ============================================================================

checkpoint_stage = checkpoint.get(
    "stage",
    "unknown",
)

checkpoint_epoch = checkpoint.get(
    "epoch",
    "unknown",
)

checkpoint_val_f1 = checkpoint.get(
    "val_macro_f1",
    "unknown",
)


print()
print("Checkpoint metadata:")
print(f"  Stage:        {checkpoint_stage}")
print(f"  Epoch:        {checkpoint_epoch}")
print(f"  Val Macro F1: {checkpoint_val_f1}")


# ============================================================================
# CLASS MAPPING
# ============================================================================

checkpoint_class_to_idx = checkpoint.get(
    "class_to_idx"
)


if not checkpoint_class_to_idx:
    print()
    print(
        "ERROR: Checkpoint does not contain "
        "'class_to_idx'."
    )
    sys.exit(1)


CLASS_NAMES = [
    name
    for name, index in sorted(
        checkpoint_class_to_idx.items(),
        key=lambda item: item[1],
    )
]


CLASS_TO_INDEX = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}


print()
print("Checkpoint class mapping:")

for index, class_name in enumerate(CLASS_NAMES):
    print(
        f"  {index:>2}: {class_name}"
    )


print()
print(
    f"Number of classifier classes: "
    f"{len(CLASS_NAMES)}"
)


# ============================================================================
# BUILD EXACT WASTEWISE MODEL
# ============================================================================

print()
print("=" * 80)
print("BUILDING WASTEWISE RESNET50")
print("=" * 80)


model = WasteResNet50(
    num_classes=len(CLASS_NAMES),
    pretrained=False,
)


print()
print(
    "[OK] WasteResNet50 architecture created."
)


# ============================================================================
# EXTRACT MODEL STATE DICT
# ============================================================================

if "model_state_dict" in checkpoint:

    state_dict = checkpoint[
        "model_state_dict"
    ]

elif "state_dict" in checkpoint:

    state_dict = checkpoint[
        "state_dict"
    ]

else:

    print()
    print(
        "ERROR: Checkpoint does not contain "
        "'model_state_dict' or 'state_dict'."
    )

    sys.exit(1)


# ============================================================================
# HANDLE DATAPARALLEL CHECKPOINTS
# ============================================================================

state_dict = {
    key.replace("module.", "", 1)
    if key.startswith("module.")
    else key: value
    for key, value in state_dict.items()
}


# ============================================================================
# LOAD TRAINED WEIGHTS
# ============================================================================

print()
print("Loading trained weights...")


try:

    model.load_state_dict(
        state_dict,
        strict=True,
    )

except RuntimeError as exc:

    print()
    print("=" * 80)
    print("MODEL LOAD ERROR")
    print("=" * 80)

    print(exc)

    sys.exit(1)


model.to(DEVICE)
model.eval()


print()
print(
    "[OK] Expanded ResNet50 Stage 2 loaded."
)


# ============================================================================
# COLLECT FULL DATASET
# ============================================================================

print()
print("=" * 80)
print("COLLECTING FULL CLASSIFICATION BENCHMARK")
print("=" * 80)


samples = []

dataset_counts = Counter()


for folder in sorted(
    DATASET_PATH.iterdir()
):

    if not folder.is_dir():
        continue


    source_class = folder.name.lower()


    if source_class not in LABEL_MAP:
        print(
            f"[WARNING] Ignoring unknown folder: "
            f"{folder.name}"
        )
        continue


    expected_class = LABEL_MAP[
        source_class
    ]


    # --------------------------------------------------------------
    # Verify mapped class exists in checkpoint.
    # --------------------------------------------------------------

    if expected_class not in CLASS_TO_INDEX:

        print(
            "[WARNING] Dataset class "
            f"'{source_class}' maps to "
            f"'{expected_class}', but this "
            "class is not in the checkpoint."
        )

        continue


    # --------------------------------------------------------------
    # Collect every supported image.
    # --------------------------------------------------------------

    images = [
        path
        for path in folder.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    ]


    images = sorted(images)


    dataset_counts[
        expected_class
    ] = len(images)


    for image_path in images:

        samples.append(
            (
                image_path,
                expected_class,
            )
        )


# ============================================================================
# DATASET SUMMARY
# ============================================================================

print()

print(
    "FULL DATASET CLASS COUNTS:"
)

print("-" * 80)


for class_name in sorted(
    dataset_counts.keys()
):

    print(
        f"  {class_name:<25} "
        f"{dataset_counts[class_name]:>6}"
    )


print("-" * 80)

print(
    f"  {'TOTAL':<25} "
    f"{len(samples):>6}"
)


print()
print(
    f"Total images collected: {len(samples)}"
)

print(
    f"Classes evaluated:      "
    f"{len(dataset_counts)}"
)


# ============================================================================
# SANITY CHECK
# ============================================================================

EXPECTED_TOTAL = 19762


if len(samples) != EXPECTED_TOTAL:

    print()
    print(
        "[WARNING] Expected approximately "
        f"{EXPECTED_TOTAL} images, but found "
        f"{len(samples)}."
    )

else:

    print()
    print(
        "[PASS] Full dataset count matches "
        f"expected {EXPECTED_TOTAL} images."
    )


# ============================================================================
# INFERENCE
# ============================================================================

print()
print("=" * 80)
print("RUNNING FULL DATASET INFERENCE")
print("=" * 80)

print()
print(
    "This will evaluate every image."
)

print()


correct = 0
total = 0

per_class_total = Counter()
per_class_correct = Counter()

predicted_classes = Counter()

misclassified = []

errors = []


# ============================================================================
# RUN MODEL
# ============================================================================

with torch.no_grad():

    for index, (
        image_path,
        expected_class,
    ) in enumerate(
        samples,
        start=1,
    ):

        try:

            # --------------------------------------------------------------
            # LOAD IMAGE
            # --------------------------------------------------------------

            image = Image.open(
                image_path
            ).convert("RGB")


            # --------------------------------------------------------------
            # EXACT PRODUCTION PREPROCESSING
            # --------------------------------------------------------------

            tensor = RESNET_TRANSFORM(
                image
            ).unsqueeze(0).to(DEVICE)


            # --------------------------------------------------------------
            # CLASSIFICATION
            # --------------------------------------------------------------

            logits = model(
                tensor
            )


            probabilities = torch.softmax(
                logits,
                dim=1,
            )


            confidence, prediction_index = (
                torch.max(
                    probabilities,
                    dim=1,
                )
            )


            predicted_index = (
                prediction_index.item()
            )


            predicted_class = (
                CLASS_NAMES[
                    predicted_index
                ]
            )


            confidence_value = (
                confidence.item()
            )


            # --------------------------------------------------------------
            # STATISTICS
            # --------------------------------------------------------------

            total += 1

            per_class_total[
                expected_class
            ] += 1


            predicted_classes[
                predicted_class
            ] += 1


            if (
                predicted_class
                == expected_class
            ):

                correct += 1

                per_class_correct[
                    expected_class
                ] += 1

            else:

                misclassified.append(
                    {
                        "file": str(
                            image_path.relative_to(
                                ROOT
                            )
                        ),
                        "expected": expected_class,
                        "predicted": predicted_class,
                        "confidence": (
                            confidence_value
                        ),
                    }
                )


            # --------------------------------------------------------------
            # PROGRESS
            # --------------------------------------------------------------

            if (
                index % 250 == 0
                or index == len(samples)
            ):

                current_accuracy = (
                    correct / total
                    if total > 0
                    else 0.0
                )

                print(
                    f"  Progress: "
                    f"{index:>6}/{len(samples)} "
                    f"| Accuracy: "
                    f"{current_accuracy * 100:>6.2f}%"
                )


        except Exception as exc:

            errors.append(
                {
                    "file": str(
                        image_path.relative_to(
                            ROOT
                        )
                    ),
                    "error": str(exc),
                }
            )


            print()
            print(
                f"[ERROR] {image_path}"
            )

            print(
                f"        {exc}"
            )


# ============================================================================
# OVERALL RESULTS
# ============================================================================

accuracy = (
    correct / total
    if total > 0
    else 0.0
)


print()
print("=" * 80)
print("FULL CLASSIFIER RESULTS")
print("=" * 80)


print()
print(
    f"Total evaluated:  {total}"
)

print(
    f"Correct:          {correct}"
)

print(
    f"Incorrect:        {total - correct}"
)

print(
    f"Inference errors: {len(errors)}"
)

print(
    f"Accuracy:         "
    f"{accuracy * 100:.2f}%"
)


# ============================================================================
# PER-CLASS ACCURACY
# ============================================================================

print()
print("PER-CLASS ACCURACY")
print("-" * 80)


for class_name in sorted(
    per_class_total.keys()
):

    class_total = (
        per_class_total[
            class_name
        ]
    )

    class_correct = (
        per_class_correct[
            class_name
        ]
    )


    class_accuracy = (
        class_correct / class_total
        if class_total > 0
        else 0.0
    )


    print(
        f"  {class_name:<25} "
        f"{class_correct:>5}/"
        f"{class_total:<5} "
        f"{class_accuracy * 100:>7.2f}%"
    )


# ============================================================================
# PREDICTED CLASS DISTRIBUTION
# ============================================================================

print()
print("PREDICTED CLASS DISTRIBUTION")
print("-" * 80)


for class_name, count in (
    predicted_classes.most_common()
):

    percentage = (
        count / total * 100
        if total > 0
        else 0.0
    )


    print(
        f"  {class_name:<25} "
        f"{count:>6} "
        f"({percentage:>6.2f}%)"
    )


# ============================================================================
# MISCLASSIFICATIONS
# ============================================================================

print()
print("MISCLASSIFIED EXAMPLES")
print("-" * 80)


if not misclassified:

    print(
        "  None."
    )


else:

    print(
        f"  Total misclassified: "
        f"{len(misclassified)}"
    )

    print()

    for item in misclassified[:50]:

        print(
            f"  Expected: "
            f"{item['expected']:<20} "
            f"Predicted: "
            f"{item['predicted']:<20} "
            f"Confidence: "
            f"{item['confidence'] * 100:>6.2f}%"
        )

        print(
            f"    {item['file']}"
        )


    if len(misclassified) > 50:

        print()
        print(
            f"  ... and "
            f"{len(misclassified) - 50} more."
        )


# ============================================================================
# INFERENCE ERRORS
# ============================================================================

if errors:

    print()
    print("INFERENCE ERRORS")
    print("-" * 80)


    for item in errors[:50]:

        print(
            f"  {item['file']}"
        )

        print(
            f"    {item['error']}"
        )


    if len(errors) > 50:

        print()
        print(
            f"  ... and "
            f"{len(errors) - 50} more."
        )


# ============================================================================
# FINAL STATUS
# ============================================================================

print()
print("=" * 80)
print("FULL CLASSIFIER BENCHMARK COMPLETE")
print("=" * 80)

print()
print(
    f"Accuracy:  "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Evaluated: "
    f"{total} images"
)

print()


if errors:

    print(
        "[WARNING] Some images produced "
        "inference errors."
    )

elif total != EXPECTED_TOTAL:

    print(
        "[WARNING] Benchmark completed, "
        "but image count differs from "
        f"expected {EXPECTED_TOTAL}."
    )

else:

    print(
        "[PASS] All 19,762 images evaluated "
        "without inference errors."
    )


print("=" * 80)