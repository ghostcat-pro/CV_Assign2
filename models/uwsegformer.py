"""
UWSegFormer: Underwater Semantic Segmentation with Transformers

Implements a Transformer-based Encoder-Decoder architecture specifically designed
for underwater semantic segmentation with channel-wise attention and multi-scale aggregation.

Architecture:
    - Encoder: ResNet or MixTransformer backbone (multi-scale feature extraction)
    - Neck: UIQA (Underwater Image Quality Assessment) - Channel-wise Attention
    - Decoder: MAA (Multi-scale Aggregation Attention)

Supported Backbones:
    - ResNet: resnet18, resnet34, resnet50, resnet101 (default, no dependencies)
    - MixTransformer: mit_b0 (standalone, no dependencies), mit_b1-b5 (requires UWSegFormer-main)
"""
from typing import List, Tuple
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import ResNetBackbone, mit_b0 as standalone_mit_b0

# Try to import MixTransformer from original implementation (optional dependency)
MIT_AVAILABLE = False
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'UWSegFormer-main'))
    from mmseg.models.backbones.mix_transformer import (
        mit_b0 as original_mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5
    )
    MIT_AVAILABLE = True
    print("MixTransformer (MiT) backbones available from UWSegFormer-main!")
except ImportError:
    # print("Original MixTransformer not available. Using standalone MIT-B0 implementation.")
    # print("To use MiT-B1 through B5 backbones, ensure UWSegFormer-main folder exists.")
    a= 0

# Always have MIT-B0 available (standalone implementation)
mit_b0 = standalone_mit_b0


class UIQAModule(nn.Module):
    """
    Underwater Image Quality Assessment Module.

    Implements global channel-wise attention across multi-scale features
    to enhance feature representations for underwater images affected by
    scattering and absorption.

    The module performs:
    1. Spatial flattening via strided convolution
    2. Global state construction by concatenating all features
    3. Channel-wise self-attention computation
    4. Feature reconstruction with residual connection

    Args:
        in_channels (List[int]): Channel dimensions for each scale [C1, C2, C3, C4]
                                 e.g., [32, 64, 160, 256] for mit_b0
        P (int): Stride for spatial flattening convolution. Default: 2
                Lower P = more spatial detail but higher memory usage
    """
    def __init__(self, in_channels: List[int] = [32, 64, 160, 256], P: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.P = P
        self.num_scales = len(in_channels)
        self.total_channels = sum(in_channels)

        # Spatial flattening convolutions (per scale)
        self.flatten_convs = nn.ModuleList([
            nn.Conv2d(c, c, kernel_size=P, stride=P, padding=0)
            for c in in_channels
        ])

        # Query projections (per scale) - project to total channel dimension
        self.query_projs = nn.ModuleList([
            nn.Linear(c, self.total_channels)
            for c in in_channels
        ])

        # Global Key and Value projections
        self.key_proj = nn.Linear(self.total_channels, self.total_channels)
        self.value_proj = nn.Linear(self.total_channels, self.total_channels)

        # Instance normalization for attention scores (per scale)
        # Note: We'll use 2d version on the attention matrix
        self.attn_norms = nn.ModuleList([
            nn.InstanceNorm2d(1) for _ in range(self.num_scales)
        ])

        # Softmax for attention
        self.softmax = nn.Softmax(dim=-1)

        # Reconstruction convolutions for upsampling back to original resolution
        # We'll use bilinear interpolation instead

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Apply channel-wise attention to multi-scale features.

        Args:
            features: List of 4 feature maps from encoder
                     [F1(B,C1,H/4,W/4), F2(B,C2,H/8,W/8),
                      F3(B,C3,H/16,W/16), F4(B,C4,H/32,W/32)]

        Returns:
            List of 4 enhanced feature maps with same shapes as input
        """
        batch_size = features[0].shape[0]

        # Step 1: Spatial Flattening
        # Apply strided convolution to flatten spatial dimensions
        flattened_features = []
        spatial_info = []  # Store original spatial dimensions for reconstruction

        for i, (feat, conv) in enumerate(zip(features, self.flatten_convs)):
            B, C, H, W = feat.shape
            spatial_info.append((H, W))

            # Apply strided convolution: (B, C, H, W) -> (B, C, H/P, W/P)
            Si = conv(feat)
            _, _, H_flat, W_flat = Si.shape
            d = H_flat * W_flat

            # Reshape to (B, C, d)
            Si = Si.reshape(B, C, d)
            flattened_features.append(Si)

        # Step 2: Global State Construction
        # Pad all features to max spatial dimension and concatenate
        max_d = max(s.shape[2] for s in flattened_features)

        padded_features = []
        for Si in flattened_features:
            d_cur = Si.shape[2]
            if d_cur < max_d:
                # Pad along spatial dimension
                padding = torch.zeros(B, Si.shape[1], max_d - d_cur,
                                    device=Si.device, dtype=Si.dtype)
                Si_padded = torch.cat([Si, padding], dim=2)
            else:
                Si_padded = Si
            padded_features.append(Si_padded)

        # Concatenate along channel dimension: (B, sum(Ci), max_d)
        S = torch.cat(padded_features, dim=1)  # (B, total_channels, max_d)

        # Step 3: Key-Value Generation (Global)
        # Transpose for linear layers: (B, max_d, total_channels)
        S_t = S.transpose(1, 2)
        K = self.key_proj(S_t)      # (B, max_d, total_channels)
        V = self.value_proj(S_t)    # (B, max_d, total_channels)

        # Step 4 & 5: Query Generation and Attention (Per-scale)
        enhanced_features = []

        for i in range(self.num_scales):
            # Get this scale's flattened feature
            Si = flattened_features[i]  # (B, Ci, di)
            Ci = Si.shape[1]
            di = Si.shape[2]

            # Query projection: (B, di, Ci) -> (B, di, total_channels)
            Si_t = Si.transpose(1, 2)
            Qi = self.query_projs[i](Si_t)  # (B, di, total_channels)

            # Compute attention scores: Qi @ K^T
            # Qi: (B, di, total_channels), K: (B, max_d, total_channels)
            # Result: (B, di, max_d)
            attn_scores = torch.matmul(Qi, K.transpose(1, 2))

            # Scale by sqrt(total_channels)
            attn_scores = attn_scores / (self.total_channels ** 0.5 + 1e-6)

            # Apply Instance Normalization
            # Reshape to (B, 1, di, max_d) for InstanceNorm2d
            attn_scores = attn_scores.unsqueeze(1)
            attn_scores = self.attn_norms[i](attn_scores)
            attn_scores = attn_scores.squeeze(1)

            # Apply Softmax: (B, di, max_d)
            attn_probs = self.softmax(attn_scores)

            # Apply attention to values: (B, di, max_d) @ (B, max_d, total_channels)
            # Result: (B, di, total_channels)
            context = torch.matmul(attn_probs, V)

            # Take only the first Ci channels (local context)
            context = context[:, :, :Ci]  # (B, di, Ci)

            # Transpose back: (B, Ci, di)
            context = context.transpose(1, 2)

            # Step 6: Reconstruction
            # Reshape to spatial format
            H_orig, W_orig = spatial_info[i]
            H_flat = H_orig // self.P
            W_flat = W_orig // self.P

            # Reshape: (B, Ci, di) -> (B, Ci, H_flat, W_flat)
            context = context.reshape(B, Ci, H_flat, W_flat)

            # Upsample to original resolution: (B, Ci, H_orig, W_orig)
            context_upsampled = F.interpolate(
                context,
                size=(H_orig, W_orig),
                mode='bilinear',
                align_corners=False
            )

            # Add residual connection
            enhanced_feat = features[i] + context_upsampled
            enhanced_features.append(enhanced_feat)

        return enhanced_features


class MAADecoder(nn.Module):
    """
    Multi-scale Aggregation Attention Decoder.

    Fuses multi-scale features using gating mechanisms and attention
    to generate accurate segmentation predictions.

    The decoder performs:
    1. Feature alignment to common spatial resolution (H/4, W/4)
    2. Channel projection to common dimension
    3. Multi-path fusion with gating
    4. Final classification

    Args:
        in_channels (List[int]): Channel dimensions for each scale [C1, C2, C3, C4]
        num_classes (int): Number of segmentation classes
        fusion_channels (int): Common channel dimension for fusion. Default: 128
        dropout (float): Dropout rate before final classification. Default: 0.1
    """
    def __init__(
        self,
        in_channels: List[int] = [32, 64, 160, 256],
        num_classes: int = 8,
        fusion_channels: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fusion_channels = fusion_channels

        # Channel projection layers (1x1 conv to common dimension)
        self.proj_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, fusion_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(fusion_channels),
                nn.ReLU(inplace=True)
            )
            for c in in_channels
        ])

        # Gating convolutions for F3 and F4
        self.gate_conv_f3 = nn.Sequential(
            nn.Conv2d(fusion_channels, fusion_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fusion_channels),
            nn.Sigmoid()
        )

        self.gate_conv_f4 = nn.Sequential(
            nn.Conv2d(fusion_channels, fusion_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fusion_channels),
            nn.Sigmoid()
        )

        # Fusion convolutions
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fusion_channels, fusion_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fusion_channels),
            nn.ReLU(inplace=True)
        )

        # Classification head
        self.dropout = nn.Dropout2d(dropout)
        self.classifier = nn.Conv2d(fusion_channels, num_classes, kernel_size=1)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Decode multi-scale features to segmentation map.

        Args:
            features: List of 4 enhanced feature maps from UIQA
                     [F'1, F'2, F'3, F'4]

        Returns:
            Segmentation logits of shape (B, num_classes, H, W)
        """
        # Step 1: Align all features to F1 resolution (H/4, W/4)
        target_size = features[0].shape[2:]  # (H/4, W/4)

        aligned_features = []
        for feat in features:
            if feat.shape[2:] != target_size:
                feat = F.interpolate(
                    feat,
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                )
            aligned_features.append(feat)

        # Step 2: Channel projection to common dimension
        F1_proj = self.proj_convs[0](aligned_features[0])  # (B, 128, H/4, W/4)
        F2_proj = self.proj_convs[1](aligned_features[1])
        F3_proj = self.proj_convs[2](aligned_features[2])
        F4_proj = self.proj_convs[3](aligned_features[3])

        # Step 3: Gating mechanisms
        gate_F3 = self.gate_conv_f3(F3_proj)  # (B, 128, H/4, W/4)
        gate_F4 = self.gate_conv_f4(F4_proj)

        # Step 4: Multi-path fusion
        # Path 1: Low-level features guided by high-level
        # Fusion1 = gate_F3 * Sigmoid(F2) * Sigmoid(F1) * (F2 + F1)
        F1_sig = torch.sigmoid(F1_proj)
        F2_sig = torch.sigmoid(F2_proj)
        Fusion1 = gate_F3 * F2_sig * F1_sig * (F2_proj + F1_proj)

        # Path 2: High-level semantic guidance
        # Fusion2 = gate_F4 * F3_proj
        Fusion2 = gate_F4 * F3_proj

        # Path 3: Direct high-level features
        # Fusion3 = F4_proj
        Fusion3 = F4_proj

        # Combine all paths
        fused = Fusion1 + Fusion2 + Fusion3

        # Additional fusion convolution
        fused = self.fusion_conv(fused)

        # Step 5: Classification
        x = self.dropout(fused)
        logits = self.classifier(x)  # (B, num_classes, H/4, W/4)

        # Upsample to original resolution (H, W)
        # Factor of 4 since features are at H/4, W/4
        logits = F.interpolate(
            logits,
            scale_factor=4,
            mode='bilinear',
            align_corners=False
        )

        return logits


class UWSegFormer(nn.Module):
    """
    UWSegFormer: Underwater Semantic Segmentation with Transformers.

    A complete encoder-decoder architecture for underwater semantic segmentation
    combining multi-scale feature extraction with UIQA attention and MAA decoder.

    Architecture:
        Input (B, 3, H, W)
          ↓
        Backbone Encoder (ResNet or MixTransformer) → [F1, F2, F3, F4] multi-scale features
          ↓
        UIQA Module → [F'1, F'2, F'3, F'4] enhanced features
          ↓
        MAA Decoder → (B, num_classes, H, W) segmentation map

    Args:
        backbone (str): Backbone architecture to use.
                       ResNet options (always available): 'resnet18', 'resnet34', 'resnet50', 'resnet101'
                       MixTransformer options: 'mit_b0' (standalone, always available)
                                              'mit_b1', 'mit_b2', 'mit_b3', 'mit_b4', 'mit_b5' (requires UWSegFormer-main)
                       Default: 'resnet50' (good balance of speed and accuracy)
        num_classes (int): Number of segmentation classes. Default: 8 (SUIM dataset)
        pretrained (bool): Whether to use ImageNet pretrained weights. Default: True
        fusion_channels (int): Channel dimension for decoder fusion. Default: 128
        uiqa_stride (int): Stride P for UIQA spatial flattening. Default: 2
        dropout (float): Dropout rate in decoder. Default: 0.1

    Example:
        >>> # Using ResNet backbone (default, no dependencies)
        >>> model = UWSegFormer(backbone='resnet50', num_classes=8)
        >>> x = torch.randn(2, 3, 256, 192)
        >>> out = model(x)
        >>> print(out.shape)  # (2, 8, 256, 192)

        >>> # Using MixTransformer backbone (requires UWSegFormer-main)
        >>> model = UWSegFormer(backbone='mit_b0', num_classes=8)
    """
    def __init__(
        self,
        backbone: str = 'resnet50',
        num_classes: int = 8,
        pretrained: bool = True,
        fusion_channels: int = 128,
        uiqa_stride: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes

        # Define backbone configurations
        # Maps backbone name to (backbone_type, variant/constructor, output_channels)
        backbone_dict = {
            # ResNet backbones (always available)
            'resnet18': ('resnet', 'resnet18', [32, 64, 160, 256]),
            'resnet34': ('resnet', 'resnet34', [32, 64, 160, 256]),
            'resnet50': ('resnet', 'resnet50', [64, 128, 320, 512]),
            'resnet101': ('resnet', 'resnet101', [64, 128, 320, 512]),
        }

        # Add standalone MIT-B0 (always available)
        backbone_dict['mit_b0'] = ('mit', mit_b0, [32, 64, 160, 256])

        # Add MixTransformer B1-B5 backbones if original implementation available
        if MIT_AVAILABLE:
            backbone_dict.update({
                'mit_b1': ('mit', mit_b1, [64, 128, 320, 512]),
                'mit_b2': ('mit', mit_b2, [64, 128, 320, 512]),
                'mit_b3': ('mit', mit_b3, [64, 128, 320, 512]),
                'mit_b4': ('mit', mit_b4, [64, 128, 320, 512]),
                'mit_b5': ('mit', mit_b5, [64, 128, 320, 512]),
            })

        if backbone not in backbone_dict:
            available_backbones = list(backbone_dict.keys())
            raise ValueError(
                f"Unknown backbone: {backbone}. "
                f"Available backbones: {available_backbones}"
            )

        backbone_type, variant, channels = backbone_dict[backbone]

        # Initialize backbone based on type
        if backbone_type == 'resnet':
            # ResNet backbone with channel adaptation
            self.backbone = ResNetBackbone(
                backbone_name=variant,
                pretrained=pretrained,
                output_channels=channels
            )
        elif backbone_type == 'mit':
            # MixTransformer backbone
            # variant is the constructor function (mit_b0, mit_b1, etc.)
            # For MIT-B0, we always use the standalone implementation
            # For MIT-B1 through B5, we check if the original implementation is available
            if backbone == 'mit_b0':
                # Use standalone implementation (always available)
                self.backbone = variant(pretrained=pretrained)
            elif not MIT_AVAILABLE:
                raise RuntimeError(
                    f"MixTransformer backbone '{backbone}' requested but not available. "
                    f"Ensure UWSegFormer-main folder exists with mmseg dependencies. "
                    f"Or use 'mit_b0' which is available standalone."
                )
            else:
                self.backbone = variant(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")

        # Initialize UIQA module
        self.uiqa = UIQAModule(in_channels=channels, P=uiqa_stride)

        # Initialize MAA decoder
        self.decoder = MAADecoder(
            in_channels=channels,
            num_classes=num_classes,
            fusion_channels=fusion_channels,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through UWSegFormer.

        Args:
            x: Input images of shape (B, 3, H, W)

        Returns:
            Segmentation logits of shape (B, num_classes, H, W)
        """
        # Extract multi-scale features from backbone
        features = self.backbone(x)  # List of 4 tensors

        # Enhance features with UIQA channel-wise attention
        enhanced_features = self.uiqa(features)

        # Decode to segmentation map
        logits = self.decoder(enhanced_features)

        return logits

    def get_param_groups(self, lr: float = 1e-4):
        """
        Get parameter groups for optimizer with different learning rates.

        Typically, the backbone uses a lower learning rate than the decoder
        when fine-tuning from pretrained weights.

        Args:
            lr: Base learning rate

        Returns:
            List of parameter groups for optimizer
        """
        return [
            {'params': self.backbone.parameters(), 'lr': lr * 0.1},
            {'params': self.uiqa.parameters(), 'lr': lr},
            {'params': self.decoder.parameters(), 'lr': lr * 10},
        ]


# Factory function for easy model creation
def get_uwsegformer(
    backbone: str = 'resnet50',
    num_classes: int = 8,
    pretrained: bool = True
) -> UWSegFormer:
    """
    Factory function to create UWSegFormer model.

    Args:
        backbone: Backbone architecture
                 ResNet: 'resnet18', 'resnet34', 'resnet50', 'resnet101'
                 MixTransformer: 'mit_b0', 'mit_b1', 'mit_b2', 'mit_b3', 'mit_b4', 'mit_b5'
        num_classes: Number of segmentation classes
        pretrained: Use ImageNet pretrained weights

    Returns:
        UWSegFormer model

    Example:
        >>> # ResNet backbone
        >>> model = get_uwsegformer(backbone='resnet50', num_classes=8)

        >>> # MixTransformer backbone (requires UWSegFormer-main)
        >>> model = get_uwsegformer(backbone='mit_b0', num_classes=8)
    """
    return UWSegFormer(
        backbone=backbone,
        num_classes=num_classes,
        pretrained=pretrained
    )


if __name__ == "__main__":
    # Simple test
    print("Testing UWSegFormer components...")

    # Test with dummy data
    batch_size = 2
    num_classes = 8
    H, W = 256, 192

    # Create dummy multi-scale features (simulating backbone output)
    F1 = torch.randn(batch_size, 32, H//4, W//4)
    F2 = torch.randn(batch_size, 64, H//8, W//8)
    F3 = torch.randn(batch_size, 160, H//16, W//16)
    F4 = torch.randn(batch_size, 256, H//32, W//32)
    features = [F1, F2, F3, F4]

    print(f"\nInput features shapes:")
    for i, f in enumerate(features):
        print(f"  F{i+1}: {f.shape}")

    # Test UIQA
    print("\nTesting UIQA...")
    uiqa = UIQAModule(in_channels=[32, 64, 160, 256])
    enhanced = uiqa(features)
    print(f"Enhanced features shapes:")
    for i, f in enumerate(enhanced):
        print(f"  F'{i+1}: {f.shape}")

    # Test MAA Decoder
    print("\nTesting MAA Decoder...")
    decoder = MAADecoder(in_channels=[32, 64, 160, 256], num_classes=num_classes)
    logits = decoder(enhanced)
    print(f"Output logits shape: {logits.shape}")

    # Test complete model (without backbone to avoid import issues)
    print("\nTesting complete forward pass...")
    print("Note: Skipping backbone test to avoid import requirements")

    print("\n✓ All component tests passed!")
