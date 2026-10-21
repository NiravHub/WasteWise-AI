from pathlib import Path
import sys
import json
import time
import io
import contextlib
import re


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from src.inference.predict import WasteWisePredictor


# ============================================================
# PATHS
# ============================================================

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

IMAGE_DIR = (
    ROOT
    / "data"
    / "external"
    / "coco"
    / "val2017"
    / "val2017"
)

OUTPUT_PATH = (
    ROOT
    / "runs"
    / "detect"
    / "integrated_coco_results.json"
)


# ============================================================
# EVALUATION SETTINGS
# ============================================================

THRESHOLDS = [0.30, 0.40, 0.50, 0.60]


# Production inference prints lines like:
#
# [WASTEWISE] Rejected detection:
# YOLO=plastic
# ResNet=None
#
# or:
#
# [WASTEWISE] Rejected detection:
# YOLO=plastic
# ResNet=glass
#
# None = OOD rejection
# class name = semantic compatibility rejection

REJECTION_PATTERN = re.compile(
    r"\[WASTEWISE\] Rejected detection:"
    r"\s+YOLO=(\S+)"
    r"\s+ResNet=(\S+)"
)


# ============================================================
# RUN ONE THRESHOLD
# ============================================================

def run_threshold(threshold, images):

    print()
    print("=" * 70)
    print(f"INTEGRATED PIPELINE - THRESHOLD {threshold:.2f}")
    print("=" * 70)

    predictor = WasteWisePredictor(
        yolo_model_path=str(MODEL_PATH),
        detection_confidence=threshold,
    )

    total_images = len(images)

    images_with_final_objects = 0
    images_without_final_objects = 0

    # IMPORTANT:
    # This is FINAL ACCEPTED object count.
    # It is NOT raw YOLO detection count.
    total_final_objects = 0

    # Rejections from production logs
    total_rejected_objects = 0
    total_ood_rejections = 0
    total_compatibility_rejections = 0

    rejection_pairs = {}

    errors = 0

    final_class_counts = {}

    start_time = time.time()

    # ========================================================
    # PROCESS ALL IMAGES
    # ========================================================

    for index, image_path in enumerate(images, start=1):

        try:

            # ------------------------------------------------
            # Capture predictor stdout so we can parse
            # [WASTEWISE] Rejected detection lines.
            # ------------------------------------------------

            captured_output = io.StringIO()

            with contextlib.redirect_stdout(captured_output):
                result = predictor.predict(str(image_path))

            predictor_output = captured_output.getvalue()

            # ------------------------------------------------
            # Re-print useful rejection diagnostics
            # ------------------------------------------------

            for line in predictor_output.splitlines():

                if "[WASTEWISE] Rejected detection:" in line:
                    print(line)

            # ------------------------------------------------
            # FINAL ACCEPTED OBJECTS
            # ------------------------------------------------

            objects = result.get("objects", [])

            accepted_count = len(objects)

            total_final_objects += accepted_count

            # ------------------------------------------------
            # COUNT REJECTIONS
            # ------------------------------------------------

            matches = REJECTION_PATTERN.findall(
                predictor_output
            )

            for detector_class, resnet_class in matches:

                total_rejected_objects += 1

                # ResNet=None means classifier/OOD rejection.
                if resnet_class == "None":

                    total_ood_rejections += 1

                # A real ResNet class means the classifier
                # produced a supported class, but it was
                # incompatible with the detector class.
                else:

                    total_compatibility_rejections += 1

                # Track exact rejection pair.
                pair = f"{detector_class}->{resnet_class}"

                rejection_pairs[pair] = (
                    rejection_pairs.get(pair, 0) + 1
                )

            # ------------------------------------------------
            # IMAGE-LEVEL COUNTS
            # ------------------------------------------------

            if objects:

                images_with_final_objects += 1

            else:

                images_without_final_objects += 1

            # ------------------------------------------------
            # FINAL CLASS COUNTS
            # ------------------------------------------------

            for obj in objects:

                predicted_class = (
                    obj.get("prediction")
                    or obj.get("class")
                    or obj.get("predicted_class")
                )

                if predicted_class:

                    final_class_counts[predicted_class] = (
                        final_class_counts.get(predicted_class, 0)
                        + 1
                    )

        except Exception as exc:

            errors += 1

            print(
                f"[ERROR] {image_path.name}: {exc}"
            )

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if index % 100 == 0:

            elapsed = time.time() - start_time

            rate = (
                index / elapsed
                if elapsed > 0
                else 0
            )

            print(
                f"Processed {index}/{total_images} "
                f"({rate:.2f} images/sec)"
            )

    # ========================================================
    # FINAL COUNTS
    # ========================================================

    elapsed = time.time() - start_time

    # --------------------------------------------------------
    # Raw YOLO detections
    #
    # Every raw YOLO detection becomes either:
    #
    #   1. accepted final object
    #   2. rejected object
    #
    # Therefore:
    #
    # RAW = ACCEPTED + REJECTED
    # --------------------------------------------------------

    raw_yolo_detections = (
        total_final_objects
        + total_rejected_objects
    )

    # --------------------------------------------------------
    # Rejection accounting
    # --------------------------------------------------------

    expected_rejections = (
        total_ood_rejections
        + total_compatibility_rejections
    )

    accounting_difference = (
        total_rejected_objects
        - expected_rejections
    )

    accounting_valid = (
        raw_yolo_detections
        == (
            total_final_objects
            + total_ood_rejections
            + total_compatibility_rejections
        )
    )

    # --------------------------------------------------------
    # Positive image rate
    # --------------------------------------------------------

    final_positive_rate = (
        images_with_final_objects / total_images
        if total_images
        else 0
    )

    # --------------------------------------------------------
    # SUMMARY JSON
    # --------------------------------------------------------

    summary = {

        "threshold": threshold,

        "total_images": total_images,

        "images_with_final_objects": (
            images_with_final_objects
        ),

        "images_without_final_objects": (
            images_without_final_objects
        ),

        "final_positive_image_rate": (
            final_positive_rate
        ),

        # Correct raw detector count
        "raw_yolo_detections": (
            raw_yolo_detections
        ),

        # Rejection breakdown
        "ood_rejections": (
            total_ood_rejections
        ),

        "compatibility_rejections": (
            total_compatibility_rejections
        ),

        "total_rejected_objects": (
            total_rejected_objects
        ),

        # Final accepted objects
        "total_final_objects": (
            total_final_objects
        ),

        # Accounting
        "expected_rejections": (
            expected_rejections
        ),

        "rejection_accounting_difference": (
            accounting_difference
        ),

        "accounting_valid": (
            accounting_valid
        ),

        # Averages
        "average_raw_yolo_detections_per_image": (
            raw_yolo_detections / total_images
            if total_images
            else 0
        ),

        "average_ood_rejections_per_image": (
            total_ood_rejections / total_images
            if total_images
            else 0
        ),

        "average_compatibility_rejections_per_image": (
            total_compatibility_rejections / total_images
            if total_images
            else 0
        ),

        "average_rejected_objects_per_image": (
            total_rejected_objects / total_images
            if total_images
            else 0
        ),

        "average_final_objects_per_image": (
            total_final_objects / total_images
            if total_images
            else 0
        ),

        "errors": errors,

        "elapsed_seconds": elapsed,

        "images_per_second": (
            total_images / elapsed
            if elapsed > 0
            else 0
        ),

        # Detailed outputs
        "final_class_counts": final_class_counts,

        "rejection_pairs": rejection_pairs,
    }

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print()
    print(f"Images:                         {total_images}")

    print(
        f"Images with final objects:     "
        f"{images_with_final_objects}"
    )

    print(
        f"Final positive image rate:     "
        f"{final_positive_rate * 100:.2f}%"
    )

    print()

    print(
        f"Raw YOLO detections:            "
        f"{raw_yolo_detections}"
    )

    print(
        f"OOD rejected:                   "
        f"{total_ood_rejections}"
    )

    print(
        f"Compatibility rejected:         "
        f"{total_compatibility_rejections}"
    )

    print(
        f"Total rejected objects:         "
        f"{total_rejected_objects}"
    )

    print(
        f"Final accepted objects:         "
        f"{total_final_objects}"
    )

    print()

    print(
        f"Expected rejected:              "
        f"{expected_rejections}"
    )

    print(
        f"Accounting difference:          "
        f"{accounting_difference}"
    )

    print(
        f"Accounting valid:               "
        f"{accounting_valid}"
    )

    print()

    print(
        f"Average raw YOLO/image:         "
        f"{summary['average_raw_yolo_detections_per_image']:.4f}"
    )

    print(
        f"Average OOD rejected/image:     "
        f"{summary['average_ood_rejections_per_image']:.4f}"
    )

    print(
        f"Average compatibility/image:    "
        f"{summary['average_compatibility_rejections_per_image']:.4f}"
    )

    print(
        f"Average rejected/image:         "
        f"{summary['average_rejected_objects_per_image']:.4f}"
    )

    print(
        f"Average final/image:            "
        f"{summary['average_final_objects_per_image']:.4f}"
    )

    print(
        f"Errors:                         "
        f"{errors}"
    )

    print(
        f"Elapsed:                        "
        f"{elapsed:.1f} sec"
    )

    # ========================================================
    # TOP REJECTION PAIRS
    # ========================================================

    if rejection_pairs:

        print()
        print("Top rejection pairs:")

        for pair, count in sorted(
            rejection_pairs.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:20]:

            print(
                f"  {pair:<35} {count}"
            )

    return summary


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("WASTEWISE AI - INTEGRATED COCO EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Discover images
    # --------------------------------------------------------

    images = sorted(
        p
        for p in IMAGE_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower()
        in {
            ".jpg",
            ".jpeg",
            ".png",
        }
    )

    print(f"Dataset: {IMAGE_DIR}")
    print(f"Images: {len(images)}")
    print(f"Thresholds: {THRESHOLDS}")
    print()

    # --------------------------------------------------------
    # Run all thresholds
    # --------------------------------------------------------

    all_results = {}

    for threshold in THRESHOLDS:

        all_results[str(threshold)] = run_threshold(
            threshold,
            images,
        )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            all_results,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL INTEGRATED COCO SUMMARY")
    print("=" * 70)

    print(
        f"{'Threshold':<12}"
        f"{'Raw YOLO':<14}"
        f"{'OOD':<12}"
        f"{'Compat':<12}"
        f"{'Rejected':<14}"
        f"{'Accepted':<14}"
        f"{'Positive %':<14}"
        f"{'Valid':<8}"
    )

    for threshold in THRESHOLDS:

        r = all_results[str(threshold)]

        print(
            f"{threshold:<12.2f}"
            f"{r['raw_yolo_detections']:<14}"
            f"{r['ood_rejections']:<12}"
            f"{r['compatibility_rejections']:<12}"
            f"{r['total_rejected_objects']:<14}"
            f"{r['total_final_objects']:<14}"
            f"{r['final_positive_image_rate'] * 100:<14.2f}"
            f"{str(r['accounting_valid']):<8}"
        )

    print()

    print(f"Saved: {OUTPUT_PATH}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()