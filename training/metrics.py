"""
Evaluation metrics for semantic segmentation
Includes IoU and F-score (Dice coefficient)
"""

import torch
import numpy as np


def calculate_iou_per_class(pred, target, num_classes):
    """
    Calculate IoU for each class
    
    Args:
        pred: Predicted class indices (N, H, W)
        target: Ground truth class indices (N, H, W)
        num_classes: Number of classes
        
    Returns:
        iou_per_class: IoU for each class (num_classes,)
    """
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    
    for cls in range(num_classes):
        pred_mask = pred == cls
        target_mask = target == cls
        
        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()
        
        if union == 0:
            ious.append(float('nan'))  # Class not present
        else:
            ious.append((intersection / union).item())
    
    return np.array(ious)


def calculate_fscore_per_class(pred, target, num_classes):
    """
    Calculate F-score (Dice coefficient) for each class
    
    F-score = 2 * (precision * recall) / (precision + recall)
            = 2 * TP / (2*TP + FP + FN)
            = 2 * intersection / (pred_area + target_area)
    
    Args:
        pred: Predicted class indices (N, H, W)
        target: Ground truth class indices (N, H, W)
        num_classes: Number of classes
        
    Returns:
        fscore_per_class: F-score for each class (num_classes,)
    """
    fscores = []
    pred = pred.view(-1)
    target = target.view(-1)
    
    for cls in range(num_classes):
        pred_mask = pred == cls
        target_mask = target == cls
        
        intersection = (pred_mask & target_mask).sum().float()
        pred_area = pred_mask.sum().float()
        target_area = target_mask.sum().float()
        
        if pred_area + target_area == 0:
            fscores.append(float('nan'))  # Class not present in either
        else:
            fscore = (2 * intersection / (pred_area + target_area)).item()
            fscores.append(fscore)
    
    return np.array(fscores)


def evaluate_model_full(model, dataloader, device, num_classes=8):
    """
    Comprehensive evaluation with IoU and F-score
    
    Args:
        model: Segmentation model
        dataloader: DataLoader with test/val data
        device: torch device
        num_classes: Number of classes
        
    Returns:
        dict with:
            - mean_iou: Mean IoU across classes (ignoring NaN)
            - iou_per_class: IoU for each class
            - mean_fscore: Mean F-score across classes (ignoring NaN)
            - fscore_per_class: F-score for each class
    """
    model.eval()
    
    all_ious = []
    all_fscores = []
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            
            # Handle DeepLabV3 output
            if isinstance(outputs, dict):
                outputs = outputs['out']
            
            preds = outputs.argmax(dim=1)
            
            # Calculate per-batch metrics
            iou = calculate_iou_per_class(preds, masks, num_classes)
            fscore = calculate_fscore_per_class(preds, masks, num_classes)
            
            all_ious.append(iou)
            all_fscores.append(fscore)
    
    # Average across batches (ignoring NaN)
    all_ious = np.array(all_ious)
    all_fscores = np.array(all_fscores)
    
    mean_iou_per_class = np.nanmean(all_ious, axis=0)
    mean_fscore_per_class = np.nanmean(all_fscores, axis=0)
    
    mean_iou = np.nanmean(mean_iou_per_class)
    mean_fscore = np.nanmean(mean_fscore_per_class)
    
    return {
        'mean_iou': mean_iou,
        'iou_per_class': mean_iou_per_class,
        'mean_fscore': mean_fscore,
        'fscore_per_class': mean_fscore_per_class
    }


if __name__ == "__main__":
    # Test the metrics
    import torch
    
    # Dummy data
    pred = torch.randint(0, 8, (2, 256, 256))
    target = torch.randint(0, 8, (2, 256, 256))
    
    iou = calculate_iou_per_class(pred, target, num_classes=8)
    fscore = calculate_fscore_per_class(pred, target, num_classes=8)
    
    print("IoU per class:", iou)
    print("F-score per class:", fscore)
    print("Mean IoU:", np.nanmean(iou))
    print("Mean F-score:", np.nanmean(fscore))
    print("\nMetrics module OK!")
