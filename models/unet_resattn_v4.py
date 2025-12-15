"""
UNet-ResAttn-V4: Enhanced architecture for underwater image segmentation
Key improvements over V3:
1. ASPP module for multi-scale context aggregation
2. CBAM (Convolutional Block Attention Module) replacing SE blocks
3. Underwater-specific color correction module
4. Deep supervision with auxiliary classifiers
5. Progressive upsampling to reduce artifacts
6. Edge enhancement for better boundary detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class UnderwaterColorCorrection(nn.Module):
    """Learnable color compensation for underwater images"""
    def __init__(self, channels=3):
        super().__init__()
        # Lightweight 1x1 convs to learn color transformations
        self.correction = nn.Sequential(
            nn.Conv2d(channels, 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        correction_map = self.correction(x)
        return x * correction_map


class CBAM(nn.Module):
    """Convolutional Block Attention Module - combines channel and spatial attention"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        # Channel attention
        self.ca_pool_max = nn.AdaptiveMaxPool2d(1)
        self.ca_pool_avg = nn.AdaptiveAvgPool2d(1)
        self.ca_fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        # Spatial attention
        self.sa_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        
    def forward(self, x):
        # Channel attention
        b, c, _, _ = x.size()
        ca_max = self.ca_fc(self.ca_pool_max(x).view(b, c))
        ca_avg = self.ca_fc(self.ca_pool_avg(x).view(b, c))
        ca = torch.sigmoid(ca_max + ca_avg).view(b, c, 1, 1)
        x = x * ca
        
        # Spatial attention
        sa_max = torch.max(x, dim=1, keepdim=True)[0]
        sa_avg = torch.mean(x, dim=1, keepdim=True)
        sa = torch.sigmoid(self.sa_conv(torch.cat([sa_max, sa_avg], dim=1)))
        return x * sa


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling for multi-scale context"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Multiple parallel atrous convolutions with different rates
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # Global pooling branch
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # Projection layer
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        size = x.shape[2:]
        p1 = self.conv1(x)
        p2 = self.conv2(x)
        p3 = self.conv3(x)
        p4 = self.conv4(x)
        p5 = F.interpolate(self.pool(x), size=size, mode='bilinear', align_corners=False)
        out = torch.cat([p1, p2, p3, p4, p5], dim=1)
        return self.project(out)


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


class EdgeEnhancement(nn.Module):
    """Edge enhancement module for better boundary detection"""
    def __init__(self, channels):
        super().__init__()
        self.edge_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, 1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.edge_conv(x)


class DecoderBlock(nn.Module):
    """Decoder block with attention gate and CBAM"""
    def __init__(self, in_channels, skip_channels, out_channels, use_cbam=True):
        super().__init__()
        
        # Attention gate
        self.attention = AttentionGate(F_g=in_channels, F_l=skip_channels, 
                                       F_int=out_channels)
        
        # Progressive upsampling (reduces checkerboard artifacts)
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # Conv block
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, 
                               kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                               kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # CBAM attention
        self.use_cbam = use_cbam
        if use_cbam:
            self.cbam = CBAM(out_channels)
        
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
        
        # CBAM attention
        if self.use_cbam:
            x = self.cbam(x)
        
        return x


class UNetResAttnV4(nn.Module):
    """
    UNet-ResAttn-V4: Enhanced for underwater segmentation
    
    Args:
        in_ch: Number of input channels (3 for RGB)
        out_ch: Number of output classes
        pretrained: Use ImageNet pre-trained weights
        deep_supervision: Enable auxiliary classifiers for deep supervision
    """
    def __init__(self, in_ch=3, out_ch=8, pretrained=True, deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision
        
        # Underwater color correction (learnable)
        self.color_correction = UnderwaterColorCorrection(in_ch)
        
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
        
        # ASPP module for multi-scale context
        self.aspp = ASPP(2048, 1024)
        
        # Bridge
        self.bridge = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        
        # Decoder with attention gates and CBAM
        self.decoder4 = DecoderBlock(1024, 1024, 512, use_cbam=True)
        self.decoder3 = DecoderBlock(512, 512, 256, use_cbam=True)
        self.decoder2 = DecoderBlock(256, 256, 128, use_cbam=True)
        self.decoder1 = DecoderBlock(128, 64, 64, use_cbam=True)
        
        # Edge enhancement
        self.edge_enhance = EdgeEnhancement(64)
        
        # Deep supervision - auxiliary classifiers
        if self.deep_supervision:
            self.aux_classifier_1 = nn.Conv2d(512, out_ch, 1)  # After decoder4
            self.aux_classifier_2 = nn.Conv2d(256, out_ch, 1)  # After decoder3
        
        # Final upsampling and classification
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.final_conv = nn.Conv2d(64, out_ch, kernel_size=1)
        
        # Initialize decoder weights
        self._init_decoder_weights()
        
    def _init_decoder_weights(self):
        """Initialize decoder layers with He initialization"""
        modules_to_init = [
            self.color_correction, self.aspp, self.bridge,
            self.decoder4, self.decoder3, self.decoder2, self.decoder1,
            self.edge_enhance, self.final_upsample, self.final_conv
        ]
        if self.deep_supervision:
            modules_to_init.extend([self.aux_classifier_1, self.aux_classifier_2])
            
        for module in modules_to_init:
            for m in module.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Color correction for underwater images
        x = self.color_correction(x)
        
        # Encoder
        enc1 = self.encoder1(x)      # 192x192x64
        enc1_pool = self.pool1(enc1) # 96x96x64
        
        enc2 = self.encoder2(enc1_pool)  # 96x96x256
        enc3 = self.encoder3(enc2)       # 48x48x512
        enc4 = self.encoder4(enc3)       # 24x24x1024
        enc5 = self.encoder5(enc4)       # 12x12x2048
        
        # ASPP for multi-scale context
        aspp_out = self.aspp(enc5)  # 12x12x1024
        
        # Bridge
        bridge = self.bridge(aspp_out)  # 12x12x1024
        
        # Decoder with attention
        dec4 = self.decoder4(bridge, enc4)  # 24x24x512
        dec3 = self.decoder3(dec4, enc3)    # 48x48x256
        dec2 = self.decoder2(dec3, enc2)    # 96x96x128
        dec1 = self.decoder1(dec2, enc1)    # 192x192x64
        
        # Edge enhancement
        edge_map = self.edge_enhance(dec1)  # 192x192x1
        
        # Final output
        out = self.final_upsample(dec1)  # 384x384x64
        out = self.final_conv(out)       # 384x384xout_ch
        
        # Return with auxiliary outputs if deep supervision is enabled
        if self.deep_supervision and self.training:
            # Upsample auxiliary outputs to match input size
            aux1 = F.interpolate(self.aux_classifier_1(dec4), size=(384, 384), 
                                mode='bilinear', align_corners=False)
            aux2 = F.interpolate(self.aux_classifier_2(dec3), size=(384, 384), 
                                mode='bilinear', align_corners=False)
            return out, aux1, aux2, edge_map
        
        return out


if __name__ == "__main__":
    # Test model
    print("Testing UNet-ResAttn-V4...")
    model = UNetResAttnV4(in_ch=3, out_ch=8, pretrained=False, deep_supervision=True)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nUNet-ResAttn-V4")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass - training mode
    model.train()
    x = torch.randn(2, 3, 384, 384)
    outputs = model(x)
    
    print(f"\nTraining Mode:")
    print(f"Input shape: {x.shape}")
    if isinstance(outputs, tuple):
        main_out, aux1, aux2, edge = outputs
        print(f"Main output shape: {main_out.shape}")
        print(f"Auxiliary output 1 shape: {aux1.shape}")
        print(f"Auxiliary output 2 shape: {aux2.shape}")
        print(f"Edge map shape: {edge.shape}")
        assert main_out.shape == (2, 8, 384, 384), f"Expected (2, 8, 384, 384), got {main_out.shape}"
        assert aux1.shape == (2, 8, 384, 384), f"Expected (2, 8, 384, 384), got {aux1.shape}"
        assert aux2.shape == (2, 8, 384, 384), f"Expected (2, 8, 384, 384), got {aux2.shape}"
        assert edge.shape == (2, 1, 192, 192), f"Expected (2, 1, 192, 192), got {edge.shape}"
    
    # Test forward pass - evaluation mode
    model.eval()
    with torch.no_grad():
        y = model(x)
    
    print(f"\nEvaluation Mode:")
    print(f"Output shape: {y.shape}")
    assert y.shape == (2, 8, 384, 384), f"Expected (2, 8, 384, 384), got {y.shape}"
    
    print("\n✓ Model architecture OK")
    print("\nKey Features:")
    print("- Underwater color correction")
    print("- ASPP multi-scale context")
    print("- CBAM attention (channel + spatial)")
    print("- Attention gates in decoder")
    print("- Deep supervision with auxiliary classifiers")
    print("- Edge enhancement module")
    print("- Progressive upsampling")
