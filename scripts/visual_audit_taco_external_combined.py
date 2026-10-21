from pathlib import Path
import random

import cv2
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "taco_external_combined"
)

IMAGE_DIR = DATASET_ROOT / "images" / "train"
LABEL_DIR = DATASET_ROOT / "labels" / "train"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "dataset_audit"
    / "taco_external_combined"
)

CLASS_NAMES = [
    "plastic",
    "glass",
    "metal",
    "paper_cardboard",
    "biological",
    "other_trash",
]

SAMPLES = 12
SEED = 42


def draw_image(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(
            f"Could not read image: {image_path}"
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    height, width = image.shape[:2]

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    if label_path.exists():
        lines = label_path.read_text(
            encoding="utf-8"
        ).splitlines()

        for line in lines:
            if not line.strip():
                continue

            parts = line.split()

            if len(parts) != 5:
                continue

            class_id = int(parts[0])

            x_center = float(parts[1])
            y_center = float(parts[2])
            box_width = float(parts[3])
            box_height = float(parts[4])

            x1 = int(
                (x_center - box_width / 2)
                * width
            )

            y1 = int(
                (y_center - box_height / 2)
                * height
            )

            x2 = int(
                (x_center + box_width / 2)
                * width
            )

            y2 = int(
                (y_center + box_height / 2)
                * height
            )

            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            x2 = max(0, min(width - 1, x2))
            y2 = max(0, min(height - 1, y2))

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (255, 255, 0),
                2,
            )

            label = CLASS_NAMES[class_id]

            cv2.putText(
                image,
                label,
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

    return image


def main():
    print("=" * 70)
    print("WasteWise AI - Combined Dataset Visual Audit")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_files = sorted(
        p
        for p in IMAGE_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower()
        in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )

    random.seed(SEED)

    selected = random.sample(
        image_files,
        min(SAMPLES, len(image_files)),
    )

    print(f"\nImages available: {len(image_files)}")
    print(f"Images selected : {len(selected)}")

    # ---------------------------------------------------------
    # Generate individual annotated images
    # ---------------------------------------------------------
    annotated_images = []

    for index, image_path in enumerate(selected, start=1):
        annotated = draw_image(image_path)

        output_path = (
            OUTPUT_DIR
            / f"sample_{index:02d}_{image_path.stem}.jpg"
        )

        cv2.imwrite(
            str(output_path),
            cv2.cvtColor(
                annotated,
                cv2.COLOR_RGB2BGR,
            ),
        )

        annotated_images.append(
            (image_path.name, annotated)
        )

        print(f"{index:02d}. {image_path.name}")

    # ---------------------------------------------------------
    # Contact sheet
    # ---------------------------------------------------------
    fig, axes = plt.subplots(
        3,
        4,
        figsize=(18, 13),
    )

    axes = axes.flatten()

    for ax, (name, image) in zip(
        axes,
        annotated_images,
    ):
        ax.imshow(image)
        ax.set_title(
            name,
            fontsize=9,
        )
        ax.axis("off")

    for ax in axes[len(annotated_images):]:
        ax.axis("off")

    plt.tight_layout()

    contact_sheet = (
        OUTPUT_DIR
        / "combined_visual_audit_contact_sheet.jpg"
    )

    plt.savefig(
        contact_sheet,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print("\nContact sheet:")
    print(contact_sheet)

    print("\nVisual audit images:")
    print(OUTPUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()