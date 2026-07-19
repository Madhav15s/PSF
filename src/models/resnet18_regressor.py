"""ResNet-18 based regressor for PSF basis-weight estimation."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18Regressor(nn.Module):
    """A lightweight ResNet-18 regressor for predicting PSF basis weights."""

    def __init__(
        self,
        num_outputs: int = 5,
        pretrained: bool = False,
        dropout: float = 0.0,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        if pretrained:
            weights = ResNet18_Weights.DEFAULT
            backbone = resnet18(weights=weights)
        else:
            backbone = resnet18(weights=None)

        if in_channels != 3:
            backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        if dropout > 0.0:
            backbone.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(backbone.fc.in_features, num_outputs),
            )
        else:
            backbone.fc = nn.Linear(backbone.fc.in_features, num_outputs)

        self.backbone = backbone

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)
