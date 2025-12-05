"""
ResNet-based multi-scale feature extractor.

Provides a drop-in replacement for MixTransformer that works with
the UIQA and MAA modules without external dependencies.
"""
from typing import List
import torch
import torch.nn as nn
import torchvision.models as models


class ResNetBackbone(nn.Module):
    """
    ResNet-based backbone for multi-scale feature extraction.

    Extracts features at 4 different scales (strides 4, 8, 16, 32)
    compatible with the UWSegFormer UIQA and MAA modules.

    Args:
        backbone_name (str): ResNet variant ('resnet18', 'resnet34', 'resnet50', 'resnet101')
        pretrained (bool): Use ImageNet pretrained weights
        output_channels (List[int]): Desired output channels for each scale.
                                     If None, uses ResNet's native channels.

    Outputs:
        List of 4 feature maps at different scales:
        - F1: stride 4  (H/4, W/4)
        - F2: stride 8  (H/8, W/8)
        - F3: stride 16 (H/16, W/16)
        - F4: stride 32 (H/32, W/32)
    """
    def __init__(
        self,
        backbone_name: str = 'resnet50',
        pretrained: bool = True,
        output_channels: List[int] = None
    ):
        super().__init__()

        # Load pretrained ResNet
        if backbone_name == 'resnet18':
            resnet = models.resnet18(pretrained=pretrained)
            native_channels = [64, 64, 128, 256, 512]
        elif backbone_name == 'resnet34':
            resnet = models.resnet34(pretrained=pretrained)
            native_channels = [64, 64, 128, 256, 512]
        elif backbone_name == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            native_channels = [64, 256, 512, 1024, 2048]
        elif backbone_name == 'resnet101':
            resnet = models.resnet101(pretrained=pretrained)
            native_channels = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")

        self.backbone_name = backbone_name

        # Extract ResNet layers
        # Initial conv + bn + relu + maxpool (stride 4)
        self.conv1 = resnet.conv1       # 7x7, stride 2
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool   # 3x3, stride 2

        # ResNet stages
        self.layer1 = resnet.layer1  # stride 4  (already downsampled by conv1+maxpool)
        self.layer2 = resnet.layer2  # stride 8
        self.layer3 = resnet.layer3  # stride 16
        self.layer4 = resnet.layer4  # stride 32

        # Channel adaptation layers (if output_channels specified)
        self.output_channels = output_channels or [
            native_channels[1],  # After layer1
            native_channels[2],  # After layer2
            native_channels[3],  # After layer3
            native_channels[4],  # After layer4
        ]

        # Create 1x1 convs to adapt channels if needed
        self.adapt_layers = nn.ModuleList()
        resnet_out_channels = [native_channels[1], native_channels[2],
                              native_channels[3], native_channels[4]]

        for resnet_ch, target_ch in zip(resnet_out_channels, self.output_channels):
            if resnet_ch != target_ch:
                # Need channel adaptation
                self.adapt_layers.append(nn.Sequential(
                    nn.Conv2d(resnet_ch, target_ch, kernel_size=1, bias=False),
                    nn.BatchNorm2d(target_ch),
                    nn.ReLU(inplace=True)
                ))
            else:
                # No adaptation needed
                self.adapt_layers.append(nn.Identity())

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass to extract multi-scale features.

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            List of 4 feature tensors:
            - features[0]: (B, C1, H/4, W/4)
            - features[1]: (B, C2, H/8, W/8)
            - features[2]: (B, C3, H/16, W/16)
            - features[3]: (B, C4, H/32, W/32)
        """
        features = []

        # Initial layers (stride 4)
        x = self.conv1(x)       # stride 2
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)     # stride 2 (total stride: 4)

        # Stage 1 (stride 4)
        x = self.layer1(x)
        f1 = self.adapt_layers[0](x)
        features.append(f1)

        # Stage 2 (stride 8)
        x = self.layer2(x)
        f2 = self.adapt_layers[1](x)
        features.append(f2)

        # Stage 3 (stride 16)
        x = self.layer3(x)
        f3 = self.adapt_layers[2](x)
        features.append(f3)

        # Stage 4 (stride 32)
        x = self.layer4(x)
        f4 = self.adapt_layers[3](x)
        features.append(f4)

        return features


def resnet_backbone(
    variant: str = 'resnet50',
    pretrained: bool = True,
    output_channels: List[int] = None
) -> ResNetBackbone:
    """
    Factory function to create ResNet backbone.

    Args:
        variant: ResNet variant ('resnet18', 'resnet34', 'resnet50', 'resnet101')
        pretrained: Use ImageNet pretrained weights
        output_channels: Desired output channels for each scale [C1, C2, C3, C4]
                        For MixTransformer-B0 compatibility, use [32, 64, 160, 256]
                        For MixTransformer-B1 compatibility, use [64, 128, 320, 512]

    Returns:
        ResNetBackbone instance

    Example:
        >>> # Create backbone compatible with MiT-B0 channel dimensions
        >>> backbone = resnet_backbone('resnet50', pretrained=True,
        ...                            output_channels=[32, 64, 160, 256])
        >>> x = torch.randn(2, 3, 256, 192)
        >>> features = backbone(x)
        >>> for i, f in enumerate(features):
        ...     print(f"F{i+1}: {f.shape}")
    """
    return ResNetBackbone(
        backbone_name=variant,
        pretrained=pretrained,
        output_channels=output_channels
    )


if __name__ == "__main__":
    # Test the backbone
    print("Testing ResNet Backbone...")

    # Test with MiT-B0 compatible channels
    backbone = resnet_backbone(
        'resnet50',
        pretrained=False,
        output_channels=[32, 64, 160, 256]
    )

    x = torch.randn(2, 3, 256, 192)
    features = backbone(x)

    print(f"\nInput shape: {x.shape}")
    print(f"Output features:")
    for i, f in enumerate(features):
        print(f"  F{i+1}: {f.shape}")

    print("\n✓ ResNet backbone test passed!")
