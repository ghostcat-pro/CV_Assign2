import torch
import torch.nn as nn

class SUIMNet(nn.Module):
    """
    Paper-inspired lightweight SUIM-Net baseline (not exact weights).
    Encoder-decoder with skip connections.
    """
    def __init__(self, in_ch=3, out_ch=8, base=32):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        self.enc1 = self._blk(in_ch, base)
        self.enc2 = self._blk(base, base*2)
        self.enc3 = self._blk(base*2, base*4)
        self.enc4 = self._blk(base*4, base*8)
        self.bottleneck = self._blk(base*8, base*16)

        self.up4 = nn.ConvTranspose2d(base*16, base*8, 2, 2)
        self.dec4 = self._blk(base*16, base*8)

        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, 2)
        self.dec3 = self._blk(base*8, base*4)

        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, 2)
        self.dec2 = self._blk(base*4, base*2)

        self.up1 = nn.ConvTranspose2d(base*2, base, 2, 2)
        self.dec1 = self._blk(base*2, base)

        self.final = nn.Conv2d(base, out_ch, 1)

    def _blk(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.final(d1)
