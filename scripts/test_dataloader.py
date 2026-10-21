import torch

from src.data.dataset import create_dataloaders


def main():

    print("=" * 70)
    print("WASTEWISE AI - DATALOADER TEST")
    print("=" * 70)

    train_loader, validation_loader, test_loader, class_names = (
        create_dataloaders(
            batch_size=32,
            num_workers=0,
        )
    )

    print("\nClasses:")

    for index, class_name in enumerate(class_names):

        print(
            f"  {index}: {class_name}"
        )

    print("\nDataset sizes:")

    print(
        f"  Train      : {len(train_loader.dataset):,}"
    )

    print(
        f"  Validation : {len(validation_loader.dataset):,}"
    )

    print(
        f"  Test       : {len(test_loader.dataset):,}"
    )

    # Get one training batch
    images, labels = next(
        iter(train_loader)
    )

    print("\nFirst training batch:")

    print(
        f"  Image tensor shape : {images.shape}"
    )

    print(
        f"  Label tensor shape : {labels.shape}"
    )

    print(
        f"  Image dtype        : {images.dtype}"
    )

    print(
        f"  Label dtype        : {labels.dtype}"
    )

    print(
        f"  Image device       : {images.device}"
    )

    print(
        f"  Pixel min          : {images.min().item():.4f}"
    )

    print(
        f"  Pixel max          : {images.max().item():.4f}"
    )

    print(
        f"  Labels             : {labels.tolist()}"
    )

    # Verify shape
    assert images.shape[1:] == (
        3,
        224,
        224,
    )

    assert labels.shape[0] == images.shape[0]

    print("\n" + "=" * 70)
    print("DATALOADER TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()