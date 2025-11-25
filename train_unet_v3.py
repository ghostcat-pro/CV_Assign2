"""
Training script for UNet-ResAttn-V3 with improvements:
1. Pre-trained ResNet-50 encoder
2. Focal Loss (better than class weights for imbalance)
3. Higher resolution (384x384)
4. SE blocks in decoder
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import numpy as np
from tqdm import tqdm
import os

from models.unet_resattn_v3 import UNetResAttnV3
from datasets.suim_dataset import SUIMDataset
from datasets.augmentations import train_transforms, val_transforms
from training.loss import DiceFocalLoss
from training.eval import evaluate_loader


def get_class_weights(train_loader, num_classes=8, device='cuda'):
    """Calculate class weights from training data"""
    print("\nCalculating class weights from training data...")
    class_counts = torch.zeros(num_classes, dtype=torch.float32)
    
    for images, masks in tqdm(train_loader, desc="Computing weights"):
        for c in range(num_classes):
            class_counts[c] += (masks == c).sum().item()
    
    # Inverse frequency weighting
    total_pixels = class_counts.sum()
    class_weights = total_pixels / (num_classes * class_counts)
    
    # Normalize to have mean = 1
    class_weights = class_weights / class_weights.mean()
    
    # Apply sqrt to reduce extreme weights
    class_weights = torch.sqrt(class_weights)
    
    print(f"Class weights: {class_weights.numpy()}")
    return class_weights.to(device)


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, args):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        optimizer.step()
        
        # Calculate IoU
        with torch.no_grad():
            preds = outputs.argmax(dim=1)
            intersection = ((preds == masks) & (masks > 0)).float().sum()
            union = ((preds > 0) | (masks > 0)).float().sum()
            iou = (intersection / (union + 1e-6)).item()
        
        running_loss += loss.item()
        running_iou += iou
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'iou': f'{iou:.4f}'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_iou = running_iou / len(train_loader)
    
    return epoch_loss, epoch_iou


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Create higher resolution transforms (384x384)
    from albumentations import Compose, Resize, HorizontalFlip, VerticalFlip, \
        ShiftScaleRotate, ColorJitter, GaussianBlur, OneOf, Normalize, RandomRotate90
    from albumentations.pytorch import ToTensorV2
    
    train_transforms_384 = Compose([
        Resize(384, 384),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomRotate90(p=0.5),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.5),
        OneOf([
            ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=1.0),
            GaussianBlur(blur_limit=(3, 7), p=1.0),
        ], p=0.5),
        Normalize(),
        ToTensorV2()
    ])
    
    val_transforms_384 = Compose([
        Resize(384, 384),
        Normalize(),
        ToTensorV2()
    ])
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = SUIMDataset('data/train.txt', transform=train_transforms_384)
    val_dataset = SUIMDataset('data/val.txt', transform=val_transforms_384)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Train: {len(train_dataset)} images, Val: {len(val_dataset)} images\n")
    
    # Initialize model
    print("Initializing UNet-ResAttn-V3 (with pre-trained ResNet-50)...")
    model = UNetResAttnV3(in_ch=3, out_ch=args.num_classes, pretrained=True).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {trainable_params:,}")
    
    # Calculate class weights for focal loss alpha
    if args.use_focal:
        class_weights = get_class_weights(train_loader, args.num_classes, device)
        # Convert to list for focal loss alpha parameter
        alpha = class_weights.cpu().numpy().tolist()
        print(f"Using Focal Loss with alpha (class weights)")
    else:
        alpha = None
        print(f"Using Focal Loss without class-specific alpha")
    
    # Loss function: Dice + Focal Loss
    criterion = DiceFocalLoss(dice_weight=0.5, alpha=alpha, gamma=args.focal_gamma)
    print(f"Loss: DiceFocalLoss (dice_weight=0.5, gamma={args.focal_gamma})")
    
    # Optimizer with different learning rates for encoder and decoder
    encoder_params = []
    decoder_params = []
    
    for name, param in model.named_parameters():
        if 'encoder' in name:
            encoder_params.append(param)
        else:
            decoder_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': encoder_params, 'lr': args.lr * 0.1},  # Lower LR for pre-trained encoder
        {'params': decoder_params, 'lr': args.lr}
    ], weight_decay=args.weight_decay)
    
    print(f"Optimizer: AdamW (encoder_lr={args.lr*0.1:.2e}, decoder_lr={args.lr:.2e})")
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
    )
    print(f"Scheduler: ReduceLROnPlateau (patience=5, factor=0.5)")
    
    # Create checkpoint directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Training loop
    best_iou = 0.0
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 70)
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_iou = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args
        )
        
        # Validate
        val_iou, val_per_class = evaluate_loader(model, val_loader, device, args.num_classes)
        
        # Update LR scheduler
        scheduler.step(val_iou)
        current_lr = optimizer.param_groups[1]['lr']  # Decoder LR
        
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} mIoU: {train_iou:.4f} | "
              f"Val mIoU: {val_iou:.4f} | "
              f"LR: {current_lr:.2e}")
        
        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_iou': best_iou,
                'args': args
            }
            save_path = 'checkpoints/unet_resattn_v3_best.pth'
            torch.save(checkpoint, save_path)
            print(f"  → Saved best model: {save_path} (mIoU: {best_iou:.4f})")
    
    print("=" * 70)
    print(f"Training complete! Best val mIoU: {best_iou:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train UNet-ResAttn-V3')
    
    # Model parameters
    parser.add_argument('--num_classes', type=int, default=8, help='Number of classes')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=6, help='Batch size (reduced for 384x384)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping')
    
    # Loss parameters
    parser.add_argument('--use_focal', action='store_true', default=True, 
                       help='Use class weights in focal loss')
    parser.add_argument('--focal_gamma', type=float, default=2.0, 
                       help='Focal loss gamma parameter')
    
    args = parser.parse_args()
    
    main(args)
