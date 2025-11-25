"""
UNet-ResAttn-V3: Improved architecture with pre-trained ResNet-50 encoder
Key improvements:
1. Pre-trained ResNet-50 encoder (ImageNet weights)
2. Lightweight Squeeze-Excitation blocks in decoder
3. Attention gates preserved from V1
4. Simpler architecture than V2 (no deep supervision, no SPP)
"""

import torch
import torch.nn as nn
import torchvision.models as models


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        
    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze: Global average pooling
        squeeze = x.view(b, c, -1).mean(dim=2)
        # Excitation: Two FC layers
        excitation = torch.relu(self.fc1(squeeze))
        excitation = torch.sigmoid(self.fc2(excitation))
        # Scale
        excitation = excitation.view(b, c, 1, 1)
        return x * excitation.expand_as(x)


class AttentionGate(nn.Module):
    """Attention gate for focusing on relevant features"""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class DecoderBlock(nn.Module):
    """Decoder block with attention gate and SE block"""
    def __init__(self, in_channels, skip_channels, out_channels, use_se=True):
        super().__init__()
        
        # Attention gate
        self.attention = AttentionGate(F_g=in_channels, F_l=skip_channels, 
                                       F_int=out_channels)
        
        # Upsampling
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, 
                                           kernel_size=2, stride=2)
        
        # Conv block
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, 
                               kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Squeeze-Excitation
        self.use_se = use_se
        if use_se:
            self.se = SqueezeExcitation(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x, skip):
        # Upsample
        x = self.upsample(x)
        
        # Apply attention to skip connection
        skip = self.attention(x, skip)
        
        # Concatenate
        x = torch.cat([x, skip], dim=1)
        
        # Convolutions
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        
        # SE block
        if self.use_se:
            x = self.se(x)
        
        return x


class UNetResAttnV3(nn.Module):
    """
    UNet with ResNet-50 encoder and attention gates
    
    Args:
        in_ch: Number of input channels (3 for RGB)
        out_ch: Number of output classes
        pretrained: Use ImageNet pre-trained weights
    """
    def __init__(self, in_ch=3, out_ch=8, pretrained=True):
        super().__init__()
        
        # Load pre-trained ResNet-50
        resnet = models.resnet50(pretrained=pretrained)
        
        # Encoder: Use ResNet-50 layers
        # Input: 384x384x3
        self.encoder1 = nn.Sequential(
            resnet.conv1,      # 192x192x64
            resnet.bn1,
            resnet.relu
        )
        self.pool1 = resnet.maxpool  # 96x96x64
        
        self.encoder2 = resnet.layer1  # 96x96x256
        self.encoder3 = resnet.layer2  # 48x48x512
        self.encoder4 = resnet.layer3  # 24x24x1024
        self.encoder5 = resnet.layer4  # 12x12x2048
        
        # Bridge
        self.bridge = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        
        # Decoder with attention gates and SE blocks
        self.decoder4 = DecoderBlock(1024, 1024, 512, use_se=True)
        self.decoder3 = DecoderBlock(512, 512, 256, use_se=True)
        self.decoder2 = DecoderBlock(256, 256, 128, use_se=True)
        self.decoder1 = DecoderBlock(128, 64, 64, use_se=True)
        
        # Final upsampling and classification
        self.final_upsample = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(64, out_ch, kernel_size=1)
        
        # Initialize decoder weights
        self._init_decoder_weights()
        
    def _init_decoder_weights(self):
        """Initialize decoder layers with He initialization"""
        for m in [self.bridge, self.decoder4, self.decoder3, self.decoder2, 
                  self.decoder1, self.final_upsample, self.final_conv]:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)      # 192x192x64
        enc1_pool = self.pool1(enc1) # 96x96x64
        
        enc2 = self.encoder2(enc1_pool)  # 96x96x256
        enc3 = self.encoder3(enc2)       # 48x48x512
        enc4 = self.encoder4(enc3)       # 24x24x1024
        enc5 = self.encoder5(enc4)       # 12x12x2048
        
        # Bridge
        bridge = self.bridge(enc5)  # 12x12x1024
        
        # Decoder with attention
        dec4 = self.decoder4(bridge, enc4)  # 24x24x512
        dec3 = self.decoder3(dec4, enc3)    # 48x48x256
        dec2 = self.decoder2(dec3, enc2)    # 96x96x128
        dec1 = self.decoder1(dec2, enc1)    # 192x192x64 (use enc1, not enc1_pool)
        
        # Final output
        out = self.final_upsample(dec1)  # 384x384x64
        out = self.final_conv(out)       # 384x384xout_ch
        
        return out


if __name__ == "__main__":
    # Test model
    model = UNetResAttnV3(in_ch=3, out_ch=8, pretrained=False)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"UNet-ResAttn-V3")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 384, 384)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    assert y.shape == (2, 8, 384, 384), f"Expected (2, 8, 384, 384), got {y.shape}"
    print("✓ Model architecture OK")
