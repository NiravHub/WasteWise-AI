import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Convolution -> BatchNorm -> ReLU -> MaxPool
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x):
        return self.block(x)


class WasteCNN(nn.Module):
    """
    Custom CNN baseline for WasteWise AI.
    """

    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )

        self.pool = nn.AdaptiveAvgPool2d(
            output_size=(1, 1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Dropout(p=0.4),

            nn.Linear(
                256,
                128,
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(p=0.3),

            nn.Linear(
                128,
                num_classes,
            ),
        )

    def forward(self, x):

        x = self.features(x)

        x = self.pool(x)

        x = self.classifier(x)

        return x


if __name__ == "__main__":

    model = WasteCNN(
        num_classes=10
    )

    dummy_input = torch.randn(
        2,
        3,
        224,
        224,
    )

    output = model(dummy_input)

    print("Model test successful.")
    print(f"Input shape : {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Total parameters: "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )