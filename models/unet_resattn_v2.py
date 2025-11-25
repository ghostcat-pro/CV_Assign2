import torch
import torch.nn as nn
import torch.nn.functional as F
from models.blocks.residual_block import ResidualBlock
from models.blocks.attention_gate import AttentionGate

class SpatialPyramidPooling(nn.Module):
    """ASPP-like module for multi-scale context"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # Different dilation rates for multi-scale features
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        # Global pooling branch
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(out_ch * 5, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        size = x.shape[2:]
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = F.interpolate(self.global_pool(x), size=size, mode='bilinear', align_corners=True)
        
        out = torch.cat([feat1, feat2, feat3, feat4, feat5], dim=1)
        return self.fusion(out)


class SqueezeExcitation(nn.Module):
    """Channel attention module"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.fc(x)


class ImprovedResidualBlock(nn.Module):
    """Enhanced residual block with SE attention"""
    def __init__(self, in_ch, out_ch, dropout=0.0, use_se=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
        # Channel attention
        self.se = SqueezeExcitation(out_ch) if use_se else nn.Identity()
        
        self.skip = nn.Identity()
        if in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.relu(out + identity)
        return out


class UNetResAttnV2(nn.Module):
    """Improved UNet with:
    - Squeeze-Excitation blocks
    - Spatial Pyramid Pooling
    - Deep supervision
    - Better regularization
    """
    def __init__(self, in_ch=3, out_ch=8, base_ch=64, deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision
        
        # Encoder with SE blocks
        self.enc1 = ImprovedResidualBlock(in_ch, base_ch, use_se=True)
        self.enc2 = ImprovedResidualBlock(base_ch, base_ch*2, dropout=0.1, use_se=True)
        self.enc3 = ImprovedResidualBlock(base_ch*2, base_ch*4, dropout=0.2, use_se=True)
        self.enc4 = ImprovedResidualBlock(base_ch*4, base_ch*8, dropout=0.3, use_se=True)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck with SPP for multi-scale context
        self.bottleneck_conv = ImprovedResidualBlock(base_ch*8, base_ch*16, dropout=0.4, use_se=True)
        self.spp = SpatialPyramidPooling(base_ch*16, base_ch*16)

        # Attention gates
        self.ag1 = AttentionGate(base_ch*8, base_ch*16)
        self.ag2 = AttentionGate(base_ch*4, base_ch*8)
        self.ag3 = AttentionGate(base_ch*2, base_ch*4)
        self.ag4 = AttentionGate(base_ch, base_ch*2)

        # Decoder
        self.up1 = nn.ConvTranspose2d(base_ch*16, base_ch*8, 2, 2)
        self.dec1 = ImprovedResidualBlock(base_ch*16, base_ch*8, dropout=0.3, use_se=True)

        self.up2 = nn.ConvTranspose2d(base_ch*8, base_ch*4, 2, 2)
        self.dec2 = ImprovedResidualBlock(base_ch*8, base_ch*4, dropout=0.2, use_se=True)

        self.up3 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, 2)
        self.dec3 = ImprovedResidualBlock(base_ch*4, base_ch*2, dropout=0.1, use_se=True)

        self.up4 = nn.ConvTranspose2d(base_ch*2, base_ch, 2, 2)
        self.dec4 = ImprovedResidualBlock(base_ch*2, base_ch, use_se=True)

        # Final prediction
        self.final = nn.Conv2d(base_ch, out_ch, 1)
        
        # Deep supervision outputs
        if deep_supervision:
            self.ds1 = nn.Conv2d(base_ch*8, out_ch, 1)
            self.ds2 = nn.Conv2d(base_ch*4, out_ch, 1)
            self.ds3 = nn.Conv2d(base_ch*2, out_ch, 1)

    def forward(self, x):
        size = x.shape[2:]
        
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck with SPP
        b = self.bottleneck_conv(self.pool(e4))
        b = self.spp(b)

        # Decoder with attention
        g1 = self.ag1(e4, b)
        d1 = self.up1(b)
        d1 = self.dec1(torch.cat([d1, g1], dim=1))

        g2 = self.ag2(e3, d1)
        d2 = self.up2(d1)
        d2 = self.dec2(torch.cat([d2, g2], dim=1))

        g3 = self.ag3(e2, d2)
        d3 = self.up3(d2)
        d3 = self.dec3(torch.cat([d3, g3], dim=1))

        g4 = self.ag4(e1, d3)
        d4 = self.up4(d3)
        d4 = self.dec4(torch.cat([d4, g4], dim=1))

        final_out = self.final(d4)
        
        # Return deep supervision outputs during training
        if self.training and self.deep_supervision:
            ds1 = F.interpolate(self.ds1(d1), size=size, mode='bilinear', align_corners=True)
            ds2 = F.interpolate(self.ds2(d2), size=size, mode='bilinear', align_corners=True)
            ds3 = F.interpolate(self.ds3(d3), size=size, mode='bilinear', align_corners=True)
            return {
                'out': final_out,
                'aux1': ds1,
                'aux2': ds2,
                'aux3': ds3
            }
        
        return final_out
