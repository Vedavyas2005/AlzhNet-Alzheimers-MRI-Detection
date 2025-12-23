# src/models/alznet.py
import torch
import torch.nn as nn

from src.config import NUM_CLASSES


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.block(x)


class AlzNet(nn.Module):
    """
    Custom CNN for Alzheimer MRI classification: NC, MCI, AD.
    Input: (batch, 1, 224, 224)
    """
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32),    # -> 32 x 112 x 112
            ConvBlock(32, 64),   # -> 64 x 56 x 56
            ConvBlock(64, 128),  # -> 128 x 28 x 28
            ConvBlock(128, 256), # -> 256 x 14 x 14
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # -> 256 x 1 x 1

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


def build_model(device: torch.device) -> nn.Module:
    model = AlzNet()
    model.to(device)
    return model
