import torch
import torch.nn as nn
from models.blocks.residual_block import ResidualBlock
from models.blocks.attention_gate import AttentionGate

class UNetResAttn(nn.Module):
    def __init__(self, in_ch=3, out_ch=8, base_ch=64):
        super().__init__()
        self.enc1 = ResidualBlock(in_ch, base_ch)
        self.enc2 = ResidualBlock(base_ch, base_ch*2)
        self.enc3 = ResidualBlock(base_ch*2, base_ch*4)
        self.enc4 = ResidualBlock(base_ch*4, base_ch*8)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ResidualBlock(base_ch*8, base_ch*16, dropout=0.3)

        self.ag1 = AttentionGate(base_ch*8, base_ch*16)
        self.ag2 = AttentionGate(base_ch*4, base_ch*8)
        self.ag3 = AttentionGate(base_ch*2, base_ch*4)
        self.ag4 = AttentionGate(base_ch, base_ch*2)

        self.up1 = nn.ConvTranspose2d(base_ch*16, base_ch*8, 2, 2)
        self.dec1 = ResidualBlock(base_ch*16, base_ch*8)

        self.up2 = nn.ConvTranspose2d(base_ch*8, base_ch*4, 2, 2)
        self.dec2 = ResidualBlock(base_ch*8, base_ch*4)

        self.up3 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, 2)
        self.dec3 = ResidualBlock(base_ch*4, base_ch*2)

        self.up4 = nn.ConvTranspose2d(base_ch*2, base_ch, 2, 2)
        self.dec4 = ResidualBlock(base_ch*2, base_ch)

        self.final = nn.Conv2d(base_ch, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

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

        return self.final(d4)
