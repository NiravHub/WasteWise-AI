from src.inference.predict import WasteWisePredictor


print("=" * 70)
print("WASTEWISE DETECTOR -> CLASSIFIER MAPPING AUDIT")
print("=" * 70)

print("\nDetector -> compatible classifier classes:")

compatibility = getattr(
    WasteWisePredictor,
    "DETECTOR_CLASS_COMPATIBILITY",
    None,
)

if compatibility is None:
    # Compatibility map may be module-level rather than class-level.
    import src.inference.predict as predict_module

    compatibility = getattr(
        predict_module,
        "DETECTOR_CLASS_COMPATIBILITY",
        None,
    )

if compatibility is None:
    raise RuntimeError(
        "DETECTOR_CLASS_COMPATIBILITY was not found."
    )

for detector_class, classifier_classes in compatibility.items():
    print(f"\n{detector_class}:")
    for classifier_class in sorted(classifier_classes):
        print(f"  -> {classifier_class}")


print("\n" + "=" * 70)
print("EXPECTED DETECTOR TAXONOMY")
print("=" * 70)

expected_detector = {
    "plastic",
    "glass",
    "metal",
    "paper_cardboard",
    "biological",
    "other_trash",
}

print("Expected:")
for name in sorted(expected_detector):
    print(f"  {name}")

actual_detector = set(compatibility.keys())

print("\nMissing:")
print(sorted(expected_detector - actual_detector))

print("\nUnexpected:")
print(sorted(actual_detector - expected_detector))


print("\n" + "=" * 70)
print("CHECK")
print("=" * 70)

if actual_detector == expected_detector:
    print("PASS: Detector taxonomy exactly matches the 6-class production taxonomy.")
else:
    print("FAIL: Detector taxonomy mismatch.")


print("\n" + "=" * 70)
print("CLASSIFIER TAXONOMY")
print("=" * 70)

expected_classifier = {
    "battery",
    "biological",
    "cable",
    "cardboard",
    "clothes",
    "electronic_device",
    "gauze",
    "general_trash",
    "glass",
    "medical_cap",
    "medical_glasses",
    "medical_gloves",
    "metal",
    "paper",
    "plastic",
    "shoe_cover",
    "shoes",
    "test_tube",
    "urine_bag",
}

mapped_classifier = set()

for classes in compatibility.values():
    mapped_classifier.update(classes)

print(f"Expected classifier classes: {len(expected_classifier)}")
print(f"Classes referenced by compatibility map: {len(mapped_classifier)}")

print("\nExpected but never referenced:")
print(sorted(expected_classifier - mapped_classifier))

print("\nUnexpected classifier references:")
print(sorted(mapped_classifier - expected_classifier))

if mapped_classifier <= expected_classifier:
    print("\nPASS: Compatibility map contains no unknown classifier classes.")
else:
    print("\nFAIL: Compatibility map references unknown classifier classes.")

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)