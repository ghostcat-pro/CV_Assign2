#!/usr/bin/env python3
"""
Train the improved UNet-ResAttn-V2 model with enhancements:
- Deep supervision
- Class-weighted loss
- Better regularization
- Learning rate warmup
"""

import torch
import argparse
from pathlib import Path
from datasets.suim_dataset import SUIMDataset
from datasets.augmentations import train_transforms, val_transforms
from torch.utils.data import DataLoader
from training.loss import DiceCELoss
from training.eval import evaluate_loader
from training.utils import save_checkpoint
from models.unet_resattn_v2 import UNetResAttnV2
from tqdm import tqdm
import numpy as np

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_class_weights(train_loader, num_classes=8, device='cuda'):
    """Calculate class weights based on inverse frequency"""
    print("Calculating class weights from training data...")
    class_counts = torch.zeros(num_classes)
    
    for _, masks in tqdm(train_loader, desc="Computing weights"):
        for c in range(num_classes):
            class_counts[c] += (masks == c).sum().item()
    
    # Inverse frequency weighting
    total = class_counts.sum()
    class_weights = total / (num_classes * class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes  # Normalize
    
    print("Class weights:", class_weights.numpy())
    return class_weights.to(device)

def train_one_epoch_ds(model, loader, optimizer, criterion, device, num_classes=8, aux_weight=0.4):
    """Training with deep supervision"""
    model.train()
    losses = []
    ious = []
    
    for imgs, masks in tqdm(loader, desc="train", leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        
        # Handle deep supervision outputs
        if isinstance(outputs, dict):
            main_loss = criterion(outputs['out'], masks)
            aux_loss = sum([criterion(outputs[k], masks) for k in ['aux1', 'aux2', 'aux3']]) / 3
            loss = main_loss + aux_weight * aux_loss
            logits = outputs['out']
        else:
            loss = criterion(outputs, masks)
            logits = outputs
        
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Compute IoU
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            iou_batch = []
            for cls in range(num_classes):
                p = preds == cls
                t = masks == cls
                inter = (p & t).sum().item()
                union = (p | t).sum().item()
                if union > 0:
                    iou_batch.append(inter / union)
            if iou_batch:
                ious.append(np.mean(iou_batch))
        
        losses.append(loss.item())
    
    return np.mean(losses), np.mean(ious)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Datasets
    print("Loading datasets...")
    train_dataset = SUIMDataset("data/train.txt", transform=train_transforms)
    val_dataset = SUIMDataset("data/val.txt", transform=val_transforms)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        drop_last=True  # For batch norm stability
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    print(f"Train: {len(train_dataset)} images, Val: {len(val_dataset)} images\n")
    
    # Model
    print(f"Initializing UNet-ResAttn-V2 (improved)...")
    model = UNetResAttnV2(
        in_ch=3, 
        out_ch=args.num_classes,
        base_ch=args.base_channels,
        deep_supervision=args.deep_supervision
    )
    model = model.to(device)
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Deep supervision: {args.deep_supervision}")
    print(f"Class weighting: {args.class_weights}\n")
    
    # Loss function with optional class weights
    if args.class_weights:
        class_weights = get_class_weights(train_loader, args.num_classes, device)
        criterion = DiceCELoss(dice_weight=0.5, class_weights=class_weights)
    else:
        criterion = DiceCELoss(dice_weight=0.5)
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Cosine annealing with warmup
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,  # Restart every 10 epochs
        T_mult=2,
        eta_min=1e-6
    )
    
    # Training loop
    best_iou = 0.0
    print(f"Starting training for {args.epochs} epochs...")
    print("=" * 70)
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_iou = train_one_epoch_ds(
            model, train_loader, optimizer, criterion, device, 
            args.num_classes, args.aux_weight
        )
        
        # Validate
        val_iou, val_per_class = evaluate_loader(model, val_loader, device, args.num_classes)
        
        # Update LR
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} mIoU: {train_iou:.4f} | "
              f"Val mIoU: {val_iou:.4f} | "
              f"LR: {current_lr:.2e}")
        
        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            save_path = f"checkpoints/unet_resattn_v2_best.pth"
            save_checkpoint(model, optimizer, epoch, best_iou, save_path)
            print(f"  → Saved best model: {save_path} (mIoU: {val_iou:.4f})")
    
    print("=" * 70)
    print(f"Training complete! Best val mIoU: {best_iou:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_classes", type=int, default=8)
    parser.add_argument("--deep_supervision", action="store_true", default=True)
    parser.add_argument("--class_weights", action="store_true", default=True)
    parser.add_argument("--aux_weight", type=float, default=0.4, 
                       help="Weight for auxiliary deep supervision losses")
    
    args = parser.parse_args()
    main(args)
