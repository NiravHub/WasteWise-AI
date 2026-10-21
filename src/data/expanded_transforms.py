from torchvision import transforms


IMAGE_SIZE = 224
RESIZE_SIZE = 256

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms():
    return transforms.Compose([
        transforms.RandomResizedCrop(
            IMAGE_SIZE,
            scale=(0.75, 1.0),
            ratio=(0.85, 1.15)
        ),

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.RandomRotation(
            degrees=12
        ),

        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.15,
            hue=0.03
        ),

        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
    ])


def get_eval_transforms():
    return transforms.Compose([
        transforms.Resize(
            RESIZE_SIZE
        ),

        transforms.CenterCrop(
            IMAGE_SIZE
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
    ])
