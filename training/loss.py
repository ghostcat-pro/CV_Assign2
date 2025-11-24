import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets, num_classes):
        probs = F.softmax(logits, dim=1)
        targets_1h = F.one_hot(targets, num_classes).permute(0,3,1,2).float()

        dims = (0,2,3)
        inter = torch.sum(probs * targets_1h, dims)
        union = torch.sum(probs + targets_1h, dims)
        dice = (2*inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class DiceCELoss(nn.Module):
    def __init__(self, dice_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        return (1-self.dice_weight)*self.ce(logits, targets) + self.dice_weight*self.dice(logits, targets, num_classes)
