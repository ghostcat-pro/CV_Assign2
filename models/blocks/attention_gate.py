import torch
import torch.nn as nn

class AttentionGate(nn.Module):
    """Attention gate to filter skip-connection features."""
    def __init__(self, x_ch, g_ch, inter_ch=None):
        super().__init__()
        inter_ch = inter_ch or x_ch // 2

        self.theta_x = nn.Conv2d(x_ch, inter_ch, 1)
        self.phi_g = nn.Conv2d(g_ch, inter_ch, 1)
        self.psi = nn.Conv2d(inter_ch, 1, 1)

        self.bn = nn.BatchNorm2d(inter_ch)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        theta = self.theta_x(x)
        phi = self.phi_g(g)
        f = self.relu(self.bn(theta + phi))
        alpha = self.sigmoid(self.psi(f))
        return x * alpha
