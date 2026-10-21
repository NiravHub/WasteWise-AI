from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"


DATASETS = {
    "garbage_external_test": {
        "type": "detection",
        "path": DATA_ROOT / "processed" / "garbage_external_wastewise",
        "description": "External garbage object detection dataset mapped to WasteWise taxonomy",
        "ground_truth": "yolo",
        "class_names": {
            0: "plastic",
            1: "glass",
            2: "metal",
            3: "paper_cardboard",
            4: "biological",
        },
        "enabled": True,
    },
    "taco_practical_test": {
        "type": "detection",
        "path": DATA_ROOT / "processed" / "taco_detection_practical",
        "images": "images/test",
        "labels": "labels/test",
        "description": "Locked TACO practical 6-class test set",
        "ground_truth": "yolo",
        "class_names": {
            0: "plastic",
            1: "glass",
            2: "metal",
            3: "paper_cardboard",
            4: "biological",
            5: "other_trash",
        },
        "enabled": True,
    },
    "garbage_classification": {
        "type": "classification",
        "path": DATA_ROOT / "raw" / "garbage-dataset",
        "description": "Original 10-class WasteWise classification dataset",
        "ground_truth": "folder_name",
        "enabled": True,
    },

    "taco_practical_test": {
        "type": "detection",
        "path": DATA_ROOT / "processed" / "taco_detection_practical",
        "images": "images/test",
        "labels": "labels/test",
        "description": "Locked TACO practical 6-class test set",
        "ground_truth": "yolo",
        "enabled": True,
    },

    "garbage_external_test": {
        "type": "detection",
        "path": DATA_ROOT / "processed" / "garbage_external_wastewise",
        "description": "External garbage object detection dataset mapped to WasteWise taxonomy",
        "ground_truth": "yolo",
        "enabled": True,
    },

    "trashnet_plus_plus": {
        "type": "detection",
        "path": DATA_ROOT / "external" / "trashnet_plus_plus",
        "description": "TrashNet++ YOLO dataset",
        "ground_truth": "yolo",
        "enabled": True,
    },

    "medical_waste": {
        "type": "classification",
        "path": DATA_ROOT / "external" / "medical_waste_4_0",
        "description": "Medical waste external evaluation data",
        "ground_truth": "folder_name",
        "enabled": True,
    },

    "coco_negative": {
        "type": "negative",
        "path": DATA_ROOT / "external" / "coco",
        "description": "COCO non-waste / negative test data",
        "ground_truth": "no_waste",
        "enabled": True,
    },
}


def get_enabled_datasets():
    """Return only datasets enabled for evaluation."""
    return {
        name: config
        for name, config in DATASETS.items()
        if config.get("enabled", False)
    }


def print_registry():
    print("=" * 80)
    print("WASTEWISE DATASET REGISTRY")
    print("=" * 80)

    for name, config in get_enabled_datasets().items():
        print(f"\n{name}")
        print(f"  Type:        {config['type']}")
        print(f"  Path:        {config['path']}")
        print(f"  Ground truth:{config['ground_truth']}")
        print(f"  Description: {config['description']}")


if __name__ == "__main__":
    print_registry()