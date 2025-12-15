"""
Main training script for SUIM segmentation models.
Run with: python main_train.py --model unet_resattn --epochs 50 --batch_size 8
"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.unet_resattn import UNetResAttn
from models.unet_resattn_v2 import UNetResAttnV2
from models.unet_resattn_v3 import UNetResAttnV3
from models.unet_resattn_v4 import UNetResAttnV4
from models.suimnet import SUIMNet
from models.deeplab_resnet import get_deeplabv3
from models.uwsegformer import UWSegFormer
from datasets.suim_dataset import SUIMDataset, CLASS_NAMES, CLASS_NAMES_MERGED
from datasets.augmentations import train_transforms, val_transforms
from training.train import train_one_epoch, validate
from training.loss import DiceCELoss, V4DeepSupervisionLoss
from training.eval import evaluate_loader
from training.utils import save_checkpoint, load_checkpoint, count_parameters
from training.device_utils import get_device

def get_model(name, num_classes=8, backbone=None):
    """Load model by name."""
    if name == "unet_resattn":
        return UNetResAttn(in_ch=3, out_ch=num_classes, base_ch=64)
    elif name == "unet_resattn_v2":
        return UNetResAttnV2(in_ch=3, out_ch=num_classes, base_ch=64, deep_supervision=True)
    elif name == "unet_resattn_v3":
        return UNetResAttnV3(in_ch=3, out_ch=num_classes, pretrained=True)
    elif name == "unet_resattn_v4":
        return UNetResAttnV4(in_ch=3, out_ch=num_classes, pretrained=True, deep_supervision=True)
    elif name == "suimnet":
        return SUIMNet(in_ch=3, out_ch=num_classes, base=32)
    elif name == "deeplabv3":
        return get_deeplabv3(num_classes=num_classes, pretrained=True)
    elif name == "uwsegformer":
        # Use specified backbone or default to resnet50
        backbone = backbone or 'resnet50'
        return UWSegFormer(backbone=backbone, num_classes=num_classes, pretrained=True)
    else:
        raise ValueError(f"Unknown model: {name}")

def main(args):
    # Device (supports CUDA, MPS, and CPU)
    device = get_device()
    
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
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=args.num_workers)
    
    print(f"Train: {len(train_dataset)} images, Val: {len(val_dataset)} images")
    
    # Model
    print(f"\nInitializing {args.model}...")
    model = get_model(args.model, num_classes=args.num_classes, backbone=args.backbone)
    model = model.to(device)
    print(f"Parameters: {count_parameters(model):,}")
    
    # Loss & Optimizer
    # Use specialized loss for V4 with deep supervision
    if args.model == "unet_resattn_v4":
        # Class weights for SUIM - boost hard classes (Diver, Plant)
        # Format: [Background, Human_diver, Aquatic_plants, Wreck, Robot, Reefs_invertebrates, Sea_floor_rocks, Fish_vertebrates]
        class_weights = torch.tensor([0.1, 2.5, 3.0, 1.5, 1.5, 1.0, 1.2, 1.0]).to(device) if args.num_classes == 8 else None
        criterion = V4DeepSupervisionLoss(aux_weight=0.4, edge_weight=0.1, alpha=class_weights, gamma=2.5)
        print("Using V4DeepSupervisionLoss with class weights and deep supervision")
    else:
        criterion = DiceCELoss(dice_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
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
    print(f"\nStarting training for {args.epochs} epochs (starting from epoch {start_epoch})...")
    print("=" * 70)
    
    for epoch in range(start_epoch, args.epochs + 1):
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
            class_mode = "6cls" if args.merge_classes else "8cls"
            aug_mode = "aug" if args.augment else "noaug"
            save_path = f"checkpoints/{args.model}_{class_mode}_{aug_mode}_best.pth"
            save_checkpoint(model, optimizer, epoch, best_iou, save_path)
            print(f" Saved best model: {save_path} (mIoU: {best_iou:.4f})")
    
    print("=" * 70)
    print(f"Training complete! Best val mIoU: {best_iou:.4f}")
    
    # Final evaluation
    print("\nFinal evaluation on validation set:")
    final_miou, per_class = evaluate_loader(model, val_loader, device, args.num_classes)
    print(f"mIoU: {final_miou:.4f}")
    print("Per-class IoU:")
    class_names = CLASS_NAMES_MERGED if args.merge_classes else CLASS_NAMES
    for i, (name, iou) in enumerate(zip(class_names, per_class)):
        print(f"  {name:20s}: {iou:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SUIM segmentation models")
    
    # Data
    parser.add_argument("--train_split", default="data/train.txt")
    parser.add_argument("--val_split", default="data/val.txt")
    parser.add_argument("--images_dir", default="data/images")
    parser.add_argument("--masks_dir", default="data/masks")
    
    # Model
    parser.add_argument("--model", choices=["unet_resattn", "unet_resattn_v2", "unet_resattn_v3", "unet_resattn_v4",
                                           "suimnet", "deeplabv3", "uwsegformer"],
                       default="unet_resattn", help="Model architecture")
    parser.add_argument("--backbone", type=str, default=None,
                       help="Backbone for uwsegformer (default: resnet50). "
                            "ResNet backbones: resnet18/34/50/101 | "
                            "MiT backbones: mit_b0 (always available), mit_b1/b2/b3/b4/b5 (if installed)")
    parser.add_argument("--merge-classes", action="store_true", default=False,
                       help="Merge background, plant, and sea_floor_rock into one class (6 classes instead of 8)")
    parser.add_argument("--num_classes", type=int, default=None, 
                       help="Number of classes (auto-set to 6 if --merge-classes, else 8)")
    
    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--augment", action="store_true", default=True, help="Use data augmentation (default: True)")
    parser.add_argument("--no-augment", dest="augment", action="store_false", help="Disable data augmentation")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Auto-set num_classes if not explicitly provided
    if args.num_classes is None:
        args.num_classes = 6 if args.merge_classes else 8
    
    main(args)
