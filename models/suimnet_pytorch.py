"""
# SUIM-Net model for underwater image segmentation
# Paper: https://arxiv.org/pdf/2004.01241.pdf
# PyTorch implementation of the Keras model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class RSB(nn.Module):
    """
    Residual Skip Block
    """
    def __init__(self, in_channels, filters, kernel_size=3, strides=1, skip=True):
        super().__init__()
        f1, f2, f3, f4 = filters
        self.skip = skip
        
        # Sub-block 1
        self.conv1 = nn.Conv2d(in_channels, f1, kernel_size=1, stride=strides)
        self.bn1 = nn.BatchNorm2d(f1, momentum=0.2)
        
        # Sub-block 2
        self.conv2 = nn.Conv2d(f1, f2, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm2d(f2, momentum=0.2)
        
        # Sub-block 3
        self.conv3 = nn.Conv2d(f2, f3, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(f3, momentum=0.2)
        
        # Skip connection
        if not skip:
            self.shortcut_conv = nn.Conv2d(in_channels, f4, kernel_size=1, stride=strides)
            self.shortcut_bn = nn.BatchNorm2d(f4, momentum=0.2)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        # Skip connection
        if self.skip:
            shortcut = x
        else:
            shortcut = self.shortcut_conv(x)
            shortcut = self.shortcut_bn(shortcut)
        
        out = out + shortcut
        out = self.relu(out)
        return out


class SUIMEncoder_RSB(nn.Module):
    """
    SUIM-Net RSB-based encoder
    """
    def __init__(self, in_channels=3):
        super().__init__()
        
        # Encoder block 1
        self.enc1 = nn.Conv2d(in_channels, 64, kernel_size=5, stride=1, padding=2)
        
        # Encoder block 2
        self.bn2 = nn.BatchNorm2d(64, momentum=0.2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.rsb2_1 = RSB(64, [64, 64, 128, 128], strides=2, skip=False)
        self.rsb2_2 = RSB(128, [64, 64, 128, 128], skip=True)
        self.rsb2_3 = RSB(128, [64, 64, 128, 128], skip=True)
        
        # Encoder block 3
        self.rsb3_1 = RSB(128, [128, 128, 256, 256], strides=2, skip=False)
        self.rsb3_2 = RSB(256, [128, 128, 256, 256], skip=True)
        self.rsb3_3 = RSB(256, [128, 128, 256, 256], skip=True)
        self.rsb3_4 = RSB(256, [128, 128, 256, 256], skip=True)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # Encoder block 1
        enc_1 = self.enc1(x)
        
        # Encoder block 2
        x = self.bn2(enc_1)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.rsb2_1(x)
        x = self.rsb2_2(x)
        enc_2 = self.rsb2_3(x)
        
        # Encoder block 3
        x = self.rsb3_1(enc_2)
        x = self.rsb3_2(x)
        x = self.rsb3_3(x)
        enc_3 = self.rsb3_4(x)
        
        return [enc_1, enc_2, enc_3]


class SUIMDecoder_RSB(nn.Module):
    """
    SUIM-Net RSB-based decoder
    """
    def __init__(self, n_classes):
        super().__init__()
        
        # Decoder block 1
        self.dec1_conv = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.dec1_bn = nn.BatchNorm2d(256, momentum=0.2)
        self.dec1_up = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec1_skip_conv = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.dec1_skip_bn = nn.BatchNorm2d(256, momentum=0.2)
        
        # Decoder block 2
        self.dec2_conv = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.dec2_bn = nn.BatchNorm2d(256, momentum=0.2)
        self.dec2_up = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec2_conv2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.dec2_bn2 = nn.BatchNorm2d(128, momentum=0.2)
        self.dec2_up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec2_skip_conv = nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=1)
        self.dec2_skip_bn = nn.BatchNorm2d(128, momentum=0.2)
        
        # Decoder block 3
        self.dec3_conv = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.dec3_bn = nn.BatchNorm2d(128)
        self.dec3_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.dec3_bn2 = nn.BatchNorm2d(64, momentum=0.2)
        
        # Output layer
        self.out_conv = nn.Conv2d(64, n_classes, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, enc_features):
        enc_1, enc_2, enc_3 = enc_features
        
        # Decoder block 1
        dec_1 = self.dec1_conv(enc_3)
        dec_1 = self.dec1_bn(dec_1)
        dec_1 = self.dec1_up(dec_1)
        
        # Process skip connection with enc_2
        enc_2_skip = self.dec1_skip_conv(enc_2)
        enc_2_skip = self.dec1_skip_bn(enc_2_skip)
        enc_2_skip = self.relu(enc_2_skip)
        
        # Match spatial dimensions
        if dec_1.shape[2:] != enc_2_skip.shape[2:]:
            dec_1 = F.interpolate(dec_1, size=enc_2_skip.shape[2:], mode='bilinear', align_corners=False)
        
        dec_1s = torch.cat([enc_2_skip, dec_1], dim=1)
        
        # Decoder block 2
        dec_2 = self.dec2_conv(dec_1s)
        dec_2 = self.dec2_bn(dec_2)
        dec_2 = self.dec2_up(dec_2)
        dec_2s = self.dec2_conv2(dec_2)
        dec_2s = self.dec2_bn2(dec_2s)
        dec_2s = self.dec2_up2(dec_2s)
        
        # Match spatial dimensions with enc_1
        if dec_2s.shape[2:] != enc_1.shape[2:]:
            dec_2s = F.interpolate(dec_2s, size=enc_1.shape[2:], mode='bilinear', align_corners=False)
        
        # Skip connection with enc_1 (take only first 64 channels and concat with dec_2s=128 channels -> 192 total)
        enc_1_cat = torch.cat([enc_1[:, :64], dec_2s], dim=1)
        enc_1_skip = self.dec2_skip_conv(enc_1_cat)
        enc_1_skip = self.dec2_skip_bn(enc_1_skip)
        enc_1_skip = self.relu(enc_1_skip)
        dec_2s_full = torch.cat([enc_1_skip, dec_2s], dim=1)
        
        # Decoder block 3
        dec_3 = self.dec3_conv(dec_2s_full)
        dec_3 = self.dec3_bn(dec_3)
        dec_3s = self.dec3_conv2(dec_3)
        dec_3s = self.dec3_bn2(dec_3s)
        
        # Output with sigmoid activation (matching paper)
        out = self.out_conv(dec_3s)
        out = self.sigmoid(out)
        return out


class SUIMEncoder_VGG(nn.Module):
    """
    SUIM-Net VGG16-based encoder
    """
    def __init__(self, pretrained=True):
        super().__init__()
        vgg = models.vgg16(pretrained=pretrained)
        
        # Extract feature layers
        self.features = vgg.features
        
        # Identify pooling layer indices
        self.pool_indices = [4, 9, 16, 23]  # block1_pool, block2_pool, block3_pool, block4_pool
    
    def forward(self, x):
        pools = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.pool_indices:
                pools.append(x)
        
        return pools  # [pool1, pool2, pool3, pool4]


class SUIMDecoder_VGG(nn.Module):
    """
    SUIM-Net VGG16-based decoder
    VGG16 pooling channels: pool1=64, pool2=128, pool3=256, pool4=512
    """
    def __init__(self, n_classes):
        super().__init__()
        
        # Decoder 1: upsample pool4 (512) and concat with pool3 (256) = 768
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.Conv2d(512 + 256, 512, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(512, momentum=0.2)
        
        # Decoder 2: upsample dec1 (512) and concat with pool2 (128) = 640
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = nn.Conv2d(512 + 128, 256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(256, momentum=0.2)
        
        # Decoder 3: upsample dec2 (256) and concat with pool1 (64) = 320
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv3 = nn.Conv2d(256 + 64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128, momentum=0.2)
        
        # Decoder 4: final upsample
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        
        # Output layer
        self.out_conv = nn.Conv2d(128, n_classes, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, pool_features):
        pool1, pool2, pool3, pool4 = pool_features
        
        # Decoder 1
        dec1 = self.up1(pool4)
        # Match spatial dimensions
        if dec1.shape[2:] != pool3.shape[2:]:
            dec1 = F.interpolate(dec1, size=pool3.shape[2:], mode='bilinear', align_corners=False)
        dec1 = torch.cat([dec1, pool3], dim=1)
        dec1 = self.conv1(dec1)
        dec1 = self.bn1(dec1)
        dec1 = self.relu(dec1)
        
        # Decoder 2
        dec2 = self.up2(dec1)
        # Match spatial dimensions
        if dec2.shape[2:] != pool2.shape[2:]:
            dec2 = F.interpolate(dec2, size=pool2.shape[2:], mode='bilinear', align_corners=False)
        dec2 = torch.cat([dec2, pool2], dim=1)
        dec2 = self.conv2(dec2)
        dec2 = self.bn2(dec2)
        dec2 = self.relu(dec2)
        
        # Decoder 3
        dec3 = self.up3(dec2)
        # Match spatial dimensions
        if dec3.shape[2:] != pool1.shape[2:]:
            dec3 = F.interpolate(dec3, size=pool1.shape[2:], mode='bilinear', align_corners=False)
        dec3 = torch.cat([dec3, pool1], dim=1)
        dec3 = self.conv3(dec3)
        dec3 = self.bn3(dec3)
        dec3 = self.relu(dec3)
        
        # Decoder 4
        dec4 = self.up4(dec3)
        
        # Output with sigmoid activation (matching paper)
        out = self.out_conv(dec4)
        out = self.sigmoid(out)
        return out


class SUIMNet(nn.Module):
    """
    SUIM-Net model (Fig. 5 in the paper)
    - base='RSB' for RSB-based encoder (Fig. 5b)
    - base='VGG' for 12-layer VGG-16 encoder (Fig. 5c)
    """
    def __init__(self, base='RSB', in_channels=3, n_classes=5, pretrained_vgg=True):
        super().__init__()
        self.base = base
        
        if base == 'RSB':
            self.encoder = SUIMEncoder_RSB(in_channels=in_channels)
            self.decoder = SUIMDecoder_RSB(n_classes=n_classes)
        elif base == 'VGG':
            self.encoder = SUIMEncoder_VGG(pretrained=pretrained_vgg)
            self.decoder = SUIMDecoder_VGG(n_classes=n_classes)
        else:
            raise ValueError(f"base must be 'RSB' or 'VGG', got {base}")
    
    def forward(self, x):
        enc_features = self.encoder(x)
        out = self.decoder(enc_features)
        return out


if __name__ == "__main__":
    # Test RSB model
    model_rsb = SUIMNet(base='RSB', in_channels=3, n_classes=5)
    x = torch.randn(2, 3, 256, 320)
    out = model_rsb(x)
    print(f"RSB model output shape: {out.shape}")
    
    # Test VGG model
    model_vgg = SUIMNet(base='VGG', in_channels=3, n_classes=5, pretrained_vgg=False)
    out = model_vgg(x)
    print(f"VGG model output shape: {out.shape}")
