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

class EdgeLearningLoss(nn.Module):
    """
    Scharr operator-based edge detection loss for semantic segmentation.

    Computes edge maps from predictions and targets using Scharr filters,
    then calculates MSE or L1 loss between the edge magnitudes.

    This loss helps the model focus on boundary regions, which is particularly
    useful for underwater images where edges may be degraded by scattering.

    Args:
        loss_type (str): Type of loss to use - 'mse' or 'l1'. Default: 'mse'
        weight (float): Weight for this loss component. Default: 1.0
    """
    def __init__(self, loss_type: str = 'mse', weight: float = 1.0):
        super().__init__()
        self.loss_type = loss_type
        self.weight = weight
        self._create_scharr_kernels()

    def _create_scharr_kernels(self):
        """Create fixed Scharr operator kernels for edge detection."""
        # Scharr kernel for X gradient (horizontal edges)
        K_x = torch.tensor([
            [[-3., 0., 3.],
             [-10., 0., 10.],
             [-3., 0., 3.]]
        ], dtype=torch.float32)

        # Scharr kernel for Y gradient (vertical edges)
        K_y = torch.tensor([
            [[-3., -10., -3.],
             [0., 0., 0.],
             [3., 10., 3.]]
        ], dtype=torch.float32)

        # Register as buffers (non-learnable parameters that move with model)
        self.register_buffer('K_x', K_x)
        self.register_buffer('K_y', K_y)

    def _compute_edges(self, tensor: torch.Tensor, is_target: bool = False) -> torch.Tensor:
        """
        Compute edge magnitude using Scharr operator.

        Args:
            tensor: Input tensor of shape (B, C, H, W) or (B, H, W)
            is_target: If True, tensor is ground truth (use argmax).
                      If False, tensor is prediction (use softmax for gradients)

        Returns:
            Edge magnitude map of shape (B, 1, H, W)
        """
        # Handle different input shapes
        if len(tensor.shape) == 4 and tensor.shape[1] > 1:
            # Multi-class logits or predictions
            if is_target:
                # Ground truth: safe to use argmax (no gradients needed)
                tensor = torch.argmax(tensor, dim=1).float().unsqueeze(1)
            else:
                # Predictions: use softmax then weighted sum to maintain gradients
                probs = F.softmax(tensor, dim=1)  # (B, C, H, W)
                # Weight each class by its index for edge detection
                class_weights = torch.arange(tensor.shape[1], device=tensor.device, dtype=torch.float32)
                class_weights = class_weights.view(1, -1, 1, 1)
                tensor = (probs * class_weights).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        elif len(tensor.shape) == 3:
            # (B, H, W) targets: add channel dimension
            tensor = tensor.unsqueeze(1).float()
        elif len(tensor.shape) == 4 and tensor.shape[1] == 1:
            # Already (B, 1, H, W)
            tensor = tensor.float()

        # Apply Scharr convolution with padding to maintain spatial dimensions
        grad_x = F.conv2d(tensor, self.K_x.unsqueeze(1), padding=1)
        grad_y = F.conv2d(tensor, self.K_y.unsqueeze(1), padding=1)

        # Compute edge magnitude: sqrt(grad_x^2 + grad_y^2)
        # Add epsilon for numerical stability and gradient flow
        edges = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

        return edges

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute edge learning loss between predictions and targets.

        Args:
            pred: Predicted logits of shape (B, C, H, W)
            target: Ground truth labels of shape (B, H, W)

        Returns:
            Scalar loss value
        """
        # Compute edge maps
        edge_pred = self._compute_edges(pred, is_target=False)  # Maintain gradients
        edge_gt = self._compute_edges(target, is_target=True)   # Ground truth

        # Compute loss between edge maps
        if self.loss_type == 'mse':
            loss = F.mse_loss(edge_pred, edge_gt)
        elif self.loss_type == 'l1':
            loss = F.l1_loss(edge_pred, edge_gt)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}. Use 'mse' or 'l1'.")

        return self.weight * loss
