"""
Training script specifically for UNet-ResAttn-V4 with optimized hyperparameters
for underwater image segmentation.

Run with: python train_unet_v4.py --epochs 50 --batch_size 6
"""
import argparse
import torch
from torch.utils.data import DataLoader

from models.unet_resattn_v4 import UNetResAttnV4
from datasets.suim_dataset import SUIMDataset, CLASS_NAMES, CLASS_NAMES_MERGED
from datasets.augmentations import train_transforms, val_transforms
from training.train import train_one_epoch, validate
from training.loss import V4DeepSupervisionLoss
from training.eval import evaluate_loader
from training.utils import save_checkpoint, load_checkpoint, count_parameters
from training.device_utils import get_device


def main(args):
    # Device
    device = get_device()
    print(f"Using device: {device}")
    
    # Datasets
    print(f"\nLoading datasets...")
    print(f"Data augmentation: {'enabled' if args.augment else 'disabled'}")
    print(f"Class mode: {'6 classes (merged)' if args.merge_classes else '8 classes (original)'}")
    
    train_dataset = SUIMDataset(
        split_file=args.train_split,
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        transform=train_transforms if args.augment else val_transforms,
        merge_classes=args.merge_classes
    )
    val_dataset = SUIMDataset(
        split_file=args.val_split,
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        transform=val_transforms,
        merge_classes=args.merge_classes
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Train: {len(train_dataset)} images, Val: {len(val_dataset)} images")
    
    # Model
    print(f"\nInitializing UNet-ResAttn-V4...")
    model = UNetResAttnV4(
        in_ch=3, 
        out_ch=args.num_classes, 
        pretrained=True, 
        deep_supervision=True
    )
    model = model.to(device)
    
    total_params = count_parameters(model)
    print(f"Parameters: {total_params:,}")
    
    # Loss with class weights optimized for SUIM
    if args.num_classes == 8:
        # 8 classes: [Background, Diver, Plant, Wreck, Robot, Reefs, Sea_floor, Fish]
        class_weights = torch.tensor([
            0.1,   # Background (very common)
            2.5,   # Human_diver (rare, important)
            3.0,   # Aquatic_plants (rare, hard to segment)
            1.5,   # Wreck (medium frequency)
            1.5,   # Robot (medium frequency)
            1.0,   # Reefs_invertebrates (common)
            1.2,   # Sea_floor_rocks (common)
            1.0    # Fish_vertebrates (common)
        ]).to(device)
    else:
        # 6 classes (merged): [Background+Plant+Sea_floor, Diver, Wreck, Robot, Reefs, Fish]
        class_weights = torch.tensor([
            0.5,   # Background+Plant+Sea_floor (merged, very common)
            3.0,   # Human_diver (rare, important)
            1.5,   # Wreck (medium frequency)
            1.5,   # Robot (medium frequency)
            1.0,   # Reefs_invertebrates (common)
            1.0    # Fish_vertebrates (common)
        ]).to(device)
    
    criterion = V4DeepSupervisionLoss(
        aux_weight=0.4,      # Weight for auxiliary outputs
        edge_weight=0.1,     # Weight for edge loss
        alpha=class_weights, # Class weights for focal loss
        gamma=2.5           # Focal loss gamma (higher = more focus on hard examples)
    )
    
    print("\nLoss configuration:")
    print(f"  - Main loss: Dice (0.4) + Focal (0.6) with class weights")
    print(f"  - Auxiliary losses: 2 additional outputs with weight 0.4")
    print(f"  - Edge enhancement: Binary CE with weight 0.1")
    print(f"  - Focal gamma: 2.5 (high focus on hard examples)")
    
    # Optimizer with weight decay for regularization
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr,
        weight_decay=1e-4
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',      # Maximize IoU
        factor=0.5,      # Reduce LR by half
        patience=5,      # Wait 5 epochs before reducing
        verbose=True,
        min_lr=1e-7
    )
    
    # Resume from checkpoint
    start_epoch = 1
    best_iou = 0.0
    
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        ckpt = load_checkpoint(model, optimizer, args.resume, device)
        start_epoch = ckpt['epoch'] + 1
        best_iou = ckpt['best_iou']
        print(f"  → Loaded epoch {ckpt['epoch']} (Best mIoU: {best_iou:.4f})")
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 80)
    
    for epoch in range(start_epoch, args.epochs + 1):
        # Train
        train_loss, train_iou = train_one_epoch(
            model, train_loader, optimizer, criterion, device, num_classes=args.num_classes
        )
        
        # Validate
        val_loss, val_iou = validate(
            model, val_loader, criterion, device, num_classes=args.num_classes
        )
        
        # Scheduler step
        scheduler.step(val_iou)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} mIoU: {train_iou:.4f} | "
              f"Val Loss: {val_loss:.4f} mIoU: {val_iou:.4f} | "
              f"LR: {current_lr:.2e}")
        
        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            class_mode = "6cls" if args.merge_classes else "8cls"
            aug_mode = "aug" if args.augment else "noaug"
            save_path = f"checkpoints/unet_resattn_v4_{class_mode}_{aug_mode}_best.pth"
            save_checkpoint(model, optimizer, epoch, best_iou, save_path)
            print(f" ★ Saved best model: {save_path} (mIoU: {best_iou:.4f})")
        
        # Save periodic checkpoint
        if epoch % 10 == 0:
            class_mode = "6cls" if args.merge_classes else "8cls"
            aug_mode = "aug" if args.augment else "noaug"
            save_path = f"checkpoints/unet_resattn_v4_{class_mode}_{aug_mode}_epoch{epoch}.pth"
            save_checkpoint(model, optimizer, epoch, val_iou, save_path)
            print(f"   Saved checkpoint: {save_path}")
    
    print("=" * 80)
    print(f"Training complete! Best val mIoU: {best_iou:.4f}")
    
    # Final evaluation
    print("\nFinal evaluation on validation set:")
    final_miou, per_class = evaluate_loader(model, val_loader, device, num_classes=args.num_classes)
    print(f"mIoU: {final_miou:.4f}")
    print("\nPer-class IoU:")
    class_names = CLASS_NAMES_MERGED if args.merge_classes else CLASS_NAMES
    for i, (name, iou) in enumerate(zip(class_names, per_class)):
        print(f"  {name:25s}: {iou:.4f} ({iou*100:.2f}%)")
    
    # Log improvements over V3
    print("\n" + "=" * 80)
    print("Expected improvements over V3:")
    print("  V3 baseline: 51.91% mIoU")
    print("  V4 target:   55-58% mIoU")
    print("  Key gains from:")
    print("    - ASPP multi-scale context: +2-3%")
    print("    - CBAM dual attention: +1-2%")
    print("    - Focal loss with class weights: +1-2%")
    print("    - Deep supervision: +0.5-1%")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train UNet-ResAttn-V4 for underwater segmentation"
    )
    
    # Data
    parser.add_argument("--train_split", default="data/train.txt")
    parser.add_argument("--val_split", default="data/val.txt")
    parser.add_argument("--images_dir", default="data/images")
    parser.add_argument("--masks_dir", default="data/masks")
    
    # Model
    parser.add_argument("--merge-classes", action="store_true", default=False,
                       help="Merge background, plant, and sea_floor into one class (6 classes instead of 8)")
    parser.add_argument("--num_classes", type=int, default=None,
                       help="Number of classes (auto-set to 6 if --merge-classes, else 8)")
    
    # Training
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=6,
                       help="Batch size (default: 6, suitable for most GPUs)")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--augment", action="store_true", default=True,
                       help="Use data augmentation (default: True)")
    parser.add_argument("--no-augment", dest="augment", action="store_false",
                       help="Disable data augmentation")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loading workers")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Auto-set num_classes if not explicitly provided
    if args.num_classes is None:
        args.num_classes = 6 if args.merge_classes else 8
    
    main(args)
