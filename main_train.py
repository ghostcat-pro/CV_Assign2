"""
Main training script for SUIM segmentation models.
Run with: python main_train.py --model unet_resattn --epochs 50 --batch_size 8
"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.unet_resattn import UNetResAttn
from models.suimnet import SUIMNet
from models.deeplab_resnet import get_deeplabv3
from datasets.suim_dataset import SUIMDataset
from datasets.augmentations import train_transforms, val_transforms
from training.train import train_one_epoch, validate
from training.loss import DiceCELoss
from training.eval import evaluate_loader
from training.utils import save_checkpoint, count_parameters

def get_model(name, num_classes=8):
    """Load model by name."""
    if name == "unet_resattn":
        return UNetResAttn(in_ch=3, out_ch=num_classes, base_ch=64)
    elif name == "suimnet":
        return SUIMNet(in_ch=3, out_ch=num_classes, base=32)
    elif name == "deeplabv3":
        return get_deeplabv3(num_classes=num_classes, pretrained=True)
    else:
        raise ValueError(f"Unknown model: {name}")

def main(args):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Datasets
    print(f"\nLoading datasets...")
    train_dataset = SUIMDataset(
        split_file=args.train_split,
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        transform=train_transforms if args.augment else val_transforms
    )
    val_dataset = SUIMDataset(
        split_file=args.val_split,
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        transform=val_transforms
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=args.num_workers)
    
    print(f"Train: {len(train_dataset)} images, Val: {len(val_dataset)} images")
    
    # Model
    print(f"\nInitializing {args.model}...")
    model = get_model(args.model, num_classes=args.num_classes)
    model = model.to(device)
    print(f"Parameters: {count_parameters(model):,}")
    
    # Loss & Optimizer
    criterion = DiceCELoss(dice_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Training loop
    best_iou = 0.0
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 70)
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_iou = train_one_epoch(
            model, train_loader, optimizer, criterion, device, args.num_classes
        )
        
        # Validate
        val_loss, val_iou = validate(
            model, val_loader, criterion, device, args.num_classes
        )
        
        # Scheduler step
        scheduler.step(val_iou)
        
        # Log
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} mIoU: {train_iou:.4f} | "
              f"Val Loss: {val_loss:.4f} mIoU: {val_iou:.4f}")
        
        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            save_path = f"checkpoints/{args.model}_{'aug' if args.augment else 'noaug'}_best.pth"
            save_checkpoint(model, optimizer, epoch, best_iou, save_path)
            print(f"  → Saved best model: {save_path} (mIoU: {best_iou:.4f})")
    
    print("=" * 70)
    print(f"Training complete! Best val mIoU: {best_iou:.4f}")
    
    # Final evaluation
    print("\nFinal evaluation on validation set:")
    final_miou, per_class = evaluate_loader(model, val_loader, device, args.num_classes)
    print(f"mIoU: {final_miou:.4f}")
    print("Per-class IoU:")
    from datasets.suim_dataset import CLASS_NAMES
    for i, (name, iou) in enumerate(zip(CLASS_NAMES, per_class)):
        print(f"  {name:20s}: {iou:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SUIM segmentation models")
    
    # Data
    parser.add_argument("--train_split", default="data/train.txt")
    parser.add_argument("--val_split", default="data/val.txt")
    parser.add_argument("--images_dir", default="data/images")
    parser.add_argument("--masks_dir", default="data/masks")
    
    # Model
    parser.add_argument("--model", choices=["unet_resattn", "suimnet", "deeplabv3"],
                       default="unet_resattn", help="Model architecture")
    parser.add_argument("--num_classes", type=int, default=8)
    
    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--augment", action="store_true", help="Use data augmentation")
    parser.add_argument("--num_workers", type=int, default=4)
    
    args = parser.parse_args()
    main(args)
