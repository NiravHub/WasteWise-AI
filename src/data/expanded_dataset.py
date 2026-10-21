from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class ExpandedWasteDataset(Dataset):

    def __init__(
        self,
        csv_path,
        transform=None,
        class_to_idx=None
    ):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        self.transform = transform

        if class_to_idx is None:
            classes = sorted(
                self.df["canonical_class"].unique()
            )

            self.class_to_idx = {
                name: idx
                for idx, name in enumerate(classes)
            }
        else:
            self.class_to_idx = class_to_idx

        self.idx_to_class = {
            idx: name
            for name, idx in self.class_to_idx.items()
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        image_path = Path(row["path"])
        label_name = row["canonical_class"]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = self.class_to_idx[label_name]

        return image, label
