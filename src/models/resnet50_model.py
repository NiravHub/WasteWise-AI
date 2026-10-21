"""
WasteWise AI - ResNet50 Transfer Learning Model

Uses ImageNet-pretrained ResNet50 as the visual feature extractor
and replaces the final classification layer for the 10 waste classes.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class WasteResNet50(nn.Module):
    """
    ResNet50 adapted for WasteWise waste classification.
    """

    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()

        if pretrained:
            weights = ResNet50_Weights.DEFAULT
        else:
            weights = None

        self.backbone = resnet50(weights=weights)

        # Number of features entering the original FC layer
        in_features = self.backbone.fc.in_features

        # Replace ImageNet's 1000-class classifier
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    def extract_features(self, x):
        """
        Extract the 2048-dimensional ResNet50 feature representation
        immediately before the custom classification head.

        These features are useful for detecting whether an input image
        resembles the known waste categories.
        """

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)

        x = torch.flatten(
            x,
            1
        )

        return x

    def freeze_backbone(self):
        """
        Freeze all ResNet layers except the custom classifier.
        """

        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def unfreeze_last_layers(self):
        """
        Unfreeze layer4 and the custom classifier for fine-tuning.
        """

        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.backbone.layer4.parameters():
            param.requires_grad = True

        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def unfreeze_all(self):
        """
        Optional full fine-tuning.
        """

        for param in self.backbone.parameters():
            param.requires_grad = True


if __name__ == "__main__":

    print("=" * 70)
    print("WASTEWISE RESNET50 MODEL TEST")
    print("=" * 70)

    model = WasteResNet50(
        num_classes=10,
        pretrained=True
    )

    x = torch.randn(
        2,
        3,
        224,
        224
    )

    with torch.no_grad():

        output = model(x)

        features = model.extract_features(x)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Input shape       : {x.shape}"
    )

    print(
        f"Output shape      : {output.shape}"
    )

    print(
        f"Feature shape     : {features.shape}"
    )

    print(
        f"Total parameters  : {total_params:,}"
    )

    print(
        f"Trainable params  : {trainable_params:,}"
    )

    print(
        "Model test: SUCCESS"
    )