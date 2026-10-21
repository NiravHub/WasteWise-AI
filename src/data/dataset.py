from pathlib import Path

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ============================================================
# WasteWise AI - Dataset and DataLoader
# ============================================================

IMAGE_SIZE = 224

IMAGENET_MEAN = (
    0.485,
    0.456,
    0.406,
)

IMAGENET_STD = (
    0.229,
    0.224,
    0.225,
)


# ------------------------------------------------------------
# Training transforms
# ------------------------------------------------------------

train_transforms = transforms.Compose(
    [
        transforms.RandomResizedCrop(
            IMAGE_SIZE,
            scale=(0.75, 1.0),
            ratio=(0.75, 1.33),
        ),

        transforms.RandomHorizontalFlip(
            p=0.5,
        ),

        transforms.RandomRotation(
            degrees=10,
        ),

        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.15,
            hue=0.03,
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ]
)


# ------------------------------------------------------------
# Validation / Test transforms
# ------------------------------------------------------------

evaluation_transforms = transforms.Compose(
    [
        transforms.Resize(
            256,
        ),

        transforms.CenterCrop(
            IMAGE_SIZE,
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ]
)


# ------------------------------------------------------------
# Waste Dataset
# ------------------------------------------------------------

class WasteDataset(Dataset):

    def __init__(
        self,
        csv_path,
        class_names,
        transform=None,
    ):

        self.csv_path = Path(csv_path)
        self.class_names = class_names
        self.transform = transform

        self.class_to_idx = {
            class_name: index
            for index, class_name in enumerate(class_names)
        }

        self.data = pd.read_csv(
            self.csv_path
        )

        if "path" not in self.data.columns:
            raise ValueError(
                "CSV must contain a 'path' column."
            )

        if "label" not in self.data.columns:
            raise ValueError(
                "CSV must contain a 'label' column."
            )

        # Validate labels
        unknown_labels = set(
            self.data["label"]
        ) - set(self.class_names)

        if unknown_labels:

            raise ValueError(
                f"Unknown labels found: "
                f"{unknown_labels}"
            )

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_path = Path(row["path"])
        label_name = row["label"]

        # Load image
        try:

            image = Image.open(
                image_path
            ).convert("RGB")

        except Exception as error:

            raise RuntimeError(
                f"Failed to load image: "
                f"{image_path}"
            ) from error

        # Apply transforms
        if self.transform is not None:

            image = self.transform(
                image
            )

        label = self.class_to_idx[
            label_name
        ]

        return image, label


# ------------------------------------------------------------
# Utility: load class names
# ------------------------------------------------------------

def load_class_names(
    class_names_path="data/splits/class_names.txt",
):

    path = Path(class_names_path)

    if not path.exists():

        raise FileNotFoundError(
            f"Class names file not found: "
            f"{path.resolve()}"
        )

    class_names = [
        line.strip()
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if not class_names:

        raise ValueError(
            "No class names found."
        )

    return class_names


# ------------------------------------------------------------
# Create DataLoaders
# ------------------------------------------------------------

def create_dataloaders(
    train_csv="data/splits/train.csv",
    validation_csv="data/splits/validation.csv",
    test_csv="data/splits/test.csv",
    class_names_path="data/splits/class_names.txt",
    batch_size=32,
    num_workers=0,
):

    class_names = load_class_names(
        class_names_path
    )

    train_dataset = WasteDataset(
        train_csv,
        class_names,
        transform=train_transforms,
    )

    validation_dataset = WasteDataset(
        validation_csv,
        class_names,
        transform=evaluation_transforms,
    )

    test_dataset = WasteDataset(
        test_csv,
        class_names,
        transform=evaluation_transforms,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return (
        train_loader,
        validation_loader,
        test_loader,
        class_names,
    )