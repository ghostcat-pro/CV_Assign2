import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, ignore_index=255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets, num_classes):
        # Create mask for valid pixels (ignore ignore_index)
        valid_mask = (targets != self.ignore_index)
        
        # Only compute loss on valid pixels
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)
        
        # Clamp targets to valid range before one-hot
        targets_clamped = targets.clone()
        targets_clamped[~valid_mask] = 0  # Set ignored pixels to 0 temporarily
        
        probs = F.softmax(logits, dim=1)
        targets_1h = F.one_hot(targets_clamped, num_classes).permute(0,3,1,2).float()
        
        # Expand valid mask to match one-hot shape (N, C, H, W)
        valid_mask_expanded = valid_mask.unsqueeze(1).float()
        
        # Apply mask to both predictions and targets
        probs = probs * valid_mask_expanded
        targets_1h = targets_1h * valid_mask_expanded

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
        ignore_index: Label to ignore (default: 255)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        
    def forward(self, logits, targets):
        """
        Args:
            logits: (N, C, H, W) - Raw network outputs
            targets: (N, H, W) - Ground truth class indices
        """
        # Create valid mask
        valid_mask = (targets != self.ignore_index)
        
        # Compute cross entropy loss (no reduction yet)
        ce_loss = F.cross_entropy(logits, targets, reduction='none', ignore_index=self.ignore_index)
        
        # Get probabilities
        probs = F.softmax(logits, dim=1)
        
        # Get probability of the true class for each pixel (more memory efficient)
        # Instead of creating full one-hot, use gather to extract class probabilities
        # Clamp targets to valid range before gather (ignore_index pixels will be handled by ce_loss)
        targets_clamped = targets.clone()
        targets_clamped[targets == self.ignore_index] = 0  # Use 0 as safe index for ignored pixels
        pt = probs.gather(1, targets_clamped.unsqueeze(1)).squeeze(1)
        pt = torch.clamp(pt, 1e-7, 1.0)  # Numerical stability
        
        # Compute focal weight: (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply focal weight to cross entropy
        focal_loss = focal_weight * ce_loss
        
        # Apply alpha weighting if specified
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                # Per-class alpha (already on correct device via register_buffer)
                # Use clamped targets to avoid indexing errors
                alpha_t = self.alpha[targets_clamped]
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

class BinaryCrossEntropyLoss(nn.Module):
    """
    Binary Cross Entropy Loss for multi-label segmentation.
    Used in SUIM-Net paper with sigmoid activation.
    Treats each class independently (multi-label, not mutually exclusive).
    
    Args:
        ignore_index: Label to ignore (default: 255)
    """
    def __init__(self, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index
        self.bce = nn.BCELoss(reduction='none')
    
    def forward(self, probs, targets):
        """
        Args:
            probs: (N, C, H, W) - Probabilities after sigmoid (0-1 range)
            targets: (N, H, W) - Ground truth class indices
        """
        # Create valid mask
        valid_mask = (targets != self.ignore_index)
        
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=probs.device)
        
        # Convert targets to one-hot (N, C, H, W)
        num_classes = probs.shape[1]
        targets_clamped = targets.clone()
        targets_clamped[~valid_mask] = 0  # Set ignored pixels to 0 temporarily
        targets_onehot = F.one_hot(targets_clamped, num_classes).permute(0, 3, 1, 2).float()
        
        # Compute BCE per pixel, per class
        bce_loss = self.bce(probs, targets_onehot)
        
        # Apply valid mask (expand to match shape)
        valid_mask_expanded = valid_mask.unsqueeze(1).expand_as(bce_loss)
        bce_loss = bce_loss * valid_mask_expanded.float()
        
        # Average over valid pixels and classes
        return bce_loss.sum() / (valid_mask_expanded.sum() + 1e-7)

class DiceCELoss(nn.Module):
    def __init__(self, dice_weight=0.5, class_weights=None, ignore_index=255):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        self.dice = DiceLoss(ignore_index=ignore_index)

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        return (1-self.dice_weight)*self.ce(logits, targets) + self.dice_weight*self.dice(logits, targets, num_classes)

class DiceFocalLoss(nn.Module):
    """
    Combination of Dice Loss and Focal Loss
    Better for handling class imbalance than DiceCE
    """
    def __init__(self, dice_weight=0.5, alpha=None, gamma=2.0, ignore_index=255):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal = FocalLoss(alpha=alpha, gamma=gamma, ignore_index=ignore_index)
        self.dice = DiceLoss(ignore_index=ignore_index)

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


class V4DeepSupervisionLoss(nn.Module):
    """
    Loss function for UNet-ResAttn-V4 with deep supervision and edge enhancement.
    
    Combines:
    - Main segmentation loss (Dice + Focal)
    - Auxiliary classifier losses (2x)
    - Edge map loss (binary cross entropy)
    
    Args:
        aux_weight: Weight for auxiliary outputs (default: 0.4)
        edge_weight: Weight for edge map loss (default: 0.1)
        alpha: Class weights for focal loss (None or list/tensor)
        gamma: Focal loss gamma parameter (default: 2.5)
    """
    def __init__(self, aux_weight=0.4, edge_weight=0.1, alpha=None, gamma=2.5):
        super().__init__()
        self.aux_weight = aux_weight
        self.edge_weight = edge_weight
        
        # Main loss: Dice + Focal (better for underwater imbalance)
        self.main_loss = DiceFocalLoss(dice_weight=0.4, alpha=alpha, gamma=gamma)
        
        # Auxiliary losses: same as main
        self.aux_loss = DiceFocalLoss(dice_weight=0.4, alpha=alpha, gamma=gamma)
        
        # Edge loss: binary cross entropy
        self.edge_loss = nn.BCELoss()
    
    def _compute_edge_target(self, masks, ignore_index=255):
        """Compute edge ground truth from segmentation masks using Sobel-like operator"""
        # Convert masks to one-hot
        b, h, w = masks.shape
        device = masks.device
        
        # Create mask for valid pixels (not ignore_index)
        valid_mask = (masks != ignore_index)
        
        # Simple edge detection: differences in neighboring pixels
        edges = torch.zeros(b, 1, h, w, device=device)
        
        # Horizontal edges (only where both pixels are valid)
        valid_h = valid_mask[:, :-1, :] & valid_mask[:, 1:, :]
        edges[:, :, :-1, :] += ((masks[:, :-1, :] != masks[:, 1:, :]) & valid_h).float().unsqueeze(1)
        # Vertical edges (only where both pixels are valid)
        valid_v = valid_mask[:, :, :-1] & valid_mask[:, :, 1:]
        edges[:, :, :, :-1] += ((masks[:, :, :-1] != masks[:, :, 1:]) & valid_v).float().unsqueeze(1)
        
        # Clamp to [0, 1]
        edges = torch.clamp(edges, 0, 1)
        
        return edges
    
    def forward(self, outputs, targets):
        """
        Args:
            outputs: Tuple of (main_out, aux1, aux2, edge_map) during training,
                    or just main_out during evaluation
            targets: Ground truth masks (B, H, W)
        """
        if isinstance(outputs, tuple):
            # Training mode with deep supervision
            main_out, aux1, aux2, edge_map = outputs
            
            # Main segmentation loss
            main_loss = self.main_loss(main_out, targets)
            
            # Auxiliary losses
            aux1_loss = self.aux_loss(aux1, targets)
            aux2_loss = self.aux_loss(aux2, targets)
            
            # Edge loss
            edge_target = self._compute_edge_target(targets)
            # Match edge_map size to edge_target
            if edge_map.shape[2:] != edge_target.shape[2:]:
                edge_target = F.interpolate(edge_target, size=edge_map.shape[2:], 
                                           mode='bilinear', align_corners=False)
            edge_loss_val = self.edge_loss(edge_map, edge_target)
            
            # Combine losses
            total_loss = (main_loss + 
                         self.aux_weight * (aux1_loss + aux2_loss) + 
                         self.edge_weight * edge_loss_val)
            
            return total_loss
        else:
            # Evaluation mode - only main output
            return self.main_loss(outputs, targets)


class UWSegFormerV2DeepSupervisionLoss(nn.Module):
    """
    Loss function for UWSegFormerV2 with deep supervision and edge enhancement.
    
    Combines:
    - Main segmentation loss (Dice + Focal)
    - Auxiliary classifier losses (2x)
    - Edge map loss (binary cross entropy)
    
    Args:
        aux_weight: Weight for auxiliary outputs (default: 0.4)
        edge_weight: Weight for edge map loss (default: 0.1)
        alpha: Class weights for focal loss (None or list/tensor)
        gamma: Focal loss gamma parameter (default: 2.5)
    """
    def __init__(self, aux_weight=0.4, edge_weight=0.1, alpha=None, gamma=2.5):
        super().__init__()
        self.aux_weight = aux_weight
        self.edge_weight = edge_weight
        
        # Main loss: Dice + Focal (better for underwater imbalance)
        self.main_loss = DiceFocalLoss(dice_weight=0.4, alpha=alpha, gamma=gamma)
        
        # Auxiliary losses: same as main
        self.aux_loss = DiceFocalLoss(dice_weight=0.4, alpha=alpha, gamma=gamma)
        
        # Edge loss: binary cross entropy
        self.edge_loss = nn.BCELoss()
    
    def _compute_edge_target(self, masks, ignore_index=255):
        """Compute edge ground truth from segmentation masks using Sobel-like operator"""
        # Convert masks to one-hot
        b, h, w = masks.shape
        device = masks.device
        
        # Create mask for valid pixels (not ignore_index)
        valid_mask = (masks != ignore_index)
        
        # Simple edge detection: differences in neighboring pixels
        edges = torch.zeros(b, 1, h, w, device=device)
        
        # Horizontal edges (only where both pixels are valid)
        valid_h = valid_mask[:, :-1, :] & valid_mask[:, 1:, :]
        edges[:, :, :-1, :] += ((masks[:, :-1, :] != masks[:, 1:, :]) & valid_h).float().unsqueeze(1)
        # Vertical edges (only where both pixels are valid)
        valid_v = valid_mask[:, :, :-1] & valid_mask[:, :, 1:]
        edges[:, :, :, :-1] += ((masks[:, :, :-1] != masks[:, :, 1:]) & valid_v).float().unsqueeze(1)
        
        # Clamp to [0, 1]
        edges = torch.clamp(edges, 0, 1)
        
        return edges
    
    def forward(self, outputs, targets):
        """
        Args:
            outputs: Tuple of (main_out, aux1, aux2, edge_map) during training,
                    or just main_out during evaluation
            targets: Ground truth masks (B, H, W)
        """
        if isinstance(outputs, tuple):
            # Training mode with deep supervision
            main_out, aux1, aux2, edge_map = outputs
            
            # Main segmentation loss
            main_loss = self.main_loss(main_out, targets)
            
            # Auxiliary losses
            aux1_loss = self.aux_loss(aux1, targets)
            aux2_loss = self.aux_loss(aux2, targets)
            
            # Edge loss
            edge_target = self._compute_edge_target(targets)
            # Match edge_map size to edge_target
            if edge_map.shape[2:] != edge_target.shape[2:]:
                edge_target = F.interpolate(edge_target, size=edge_map.shape[2:], 
                                           mode='bilinear', align_corners=False)
            edge_loss_val = self.edge_loss(edge_map, edge_target)
            
            # Combine losses
            total_loss = (main_loss + 
                         self.aux_weight * (aux1_loss + aux2_loss) + 
                         self.edge_weight * edge_loss_val)
            
            return total_loss
        else:
            # Evaluation mode - only main output
            return self.main_loss(outputs, targets)
