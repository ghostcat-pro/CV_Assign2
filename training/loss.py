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

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    Paper: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    
    Args:
        alpha: Weighting factor in [0, 1] to balance positive/negative examples
               Can also be a list of weights per class
        gamma: Focusing parameter for modulating loss (default: 2.0)
               Higher gamma increases focus on hard examples
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, logits, targets):
        """
        Args:
            logits: (N, C, H, W) - Raw network outputs
            targets: (N, H, W) - Ground truth class indices
        """
        # Compute cross entropy loss (no reduction yet)
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # Get probabilities
        probs = F.softmax(logits, dim=1)
        
        # Get probability of the true class for each pixel
        targets_one_hot = F.one_hot(targets, num_classes=logits.shape[1])
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        pt = (probs * targets_one_hot).sum(dim=1)
        
        # Compute focal weight: (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply focal weight to cross entropy
        focal_loss = focal_weight * ce_loss
        
        # Apply alpha weighting if specified
        if self.alpha is not None:
            if isinstance(self.alpha, (list, torch.Tensor)):
                # Per-class alpha
                alpha_t = torch.tensor(self.alpha, device=logits.device, dtype=torch.float32)
                alpha_t = alpha_t[targets]
                focal_loss = alpha_t * focal_loss
            else:
                # Single alpha value
                focal_loss = self.alpha * focal_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DiceCELoss(nn.Module):
    def __init__(self, dice_weight=0.5, class_weights=None):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        return (1-self.dice_weight)*self.ce(logits, targets) + self.dice_weight*self.dice(logits, targets, num_classes)

class DiceFocalLoss(nn.Module):
    """
    Combination of Dice Loss and Focal Loss
    Better for handling class imbalance than DiceCE
    """
    def __init__(self, dice_weight=0.5, alpha=None, gamma=2.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        focal_loss = self.focal(logits, targets)
        dice_loss = self.dice(logits, targets, num_classes)
        return (1-self.dice_weight) * focal_loss + self.dice_weight * dice_loss
