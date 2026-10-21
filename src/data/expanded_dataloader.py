from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.expanded_dataset import ExpandedWasteDataset
from src.data.expanded_transforms import (
    get_train_transforms,
    get_eval_transforms,
)


def build_class_mapping(csv_path):
    df = pd.read_csv(csv_path)

    classes = sorted(
        df["canonical_class"].unique()
    )

    return {
        name: idx
        for idx, name in enumerate(classes)
    }


def calculate_class_weights(csv_path, class_to_idx):
    df = pd.read_csv(csv_path)

    counts = (
        df["canonical_class"]
        .value_counts()
    )

    weights = torch.ones(
        len(class_to_idx),
        dtype=torch.float32
    )

    total = len(df)
    num_classes = len(class_to_idx)

    for class_name, idx in class_to_idx.items():
        count = counts[class_name]

        weights[idx] = total / (
            num_classes * count
        )

    return weights


def build_dataloaders(
    data_dir="data/splits_expanded",
    batch_size=32,
    num_workers=4
):
    data_dir = Path(data_dir)

    train_csv = data_dir / "train.csv"
    val_csv = data_dir / "validation.csv"
    test_csv = data_dir / "test.csv"

    class_to_idx = build_class_mapping(
        train_csv
    )

    train_dataset = ExpandedWasteDataset(
        train_csv,
        transform=get_train_transforms(),
        class_to_idx=class_to_idx
    )

    validation_dataset = ExpandedWasteDataset(
        val_csv,
        transform=get_eval_transforms(),
        class_to_idx=class_to_idx
    )

    test_dataset = ExpandedWasteDataset(
        test_csv,
        transform=get_eval_transforms(),
        class_to_idx=class_to_idx
    )

    class_weights = calculate_class_weights(
        train_csv,
        class_to_idx
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )

    return (
        train_loader,
        validation_loader,
        test_loader,
        class_to_idx,
        class_weights
    )
