"""
Main training script for SUIM segmentation models.
Run with: python main_train.py --model unet_resattn --epochs 50 --batch_size 8
"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os

from models.unet_resattn import UNetResAttn
from models.unet_resattn_v2 import UNetResAttnV2
from models.unet_resattn_v3 import UNetResAttnV3
from models.unet_resattn_v4 import UNetResAttnV4
from models.suimnet_pytorch import SUIMNet
from models.deeplab_resnet import get_deeplabv3
from models.uwsegformer import UWSegFormer
# from models.uwsegformer_v2 import UWSegFormerV2  # Not implemented yet
from datasets.suim_dataset import SUIMDataset, CLASS_NAMES, CLASS_NAMES_MERGED
from datasets.augmentations import train_transforms, val_transforms
from training.train import train_one_epoch, validate
from training.loss import DiceCELoss, V4DeepSupervisionLoss, UWSegFormerV2DeepSupervisionLoss, BinaryCrossEntropyLoss
from training.eval import evaluate_loader
from training.utils import save_checkpoint, load_checkpoint, count_parameters
from training.device_utils import get_device
# from training.visualization import TrainingPlotter  # Requires matplotlib

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
        # Use backbone to select RSB or VGG (default: RSB)
        base = 'VGG' if backbone and backbone.upper() == 'VGG' else 'RSB'
        pretrained_vgg = True if base == 'VGG' else False
        return SUIMNet(base=base, in_channels=3, n_classes=num_classes, pretrained_vgg=pretrained_vgg)
    elif name == "suimnet_keras":
        # Return a flag to use Keras training pipeline
        return "KERAS_MODEL"
    elif name == "deeplabv3":
        return get_deeplabv3(num_classes=num_classes, pretrained=True)
    elif name == "uwsegformer":
        # Use specified backbone or default to resnet50
        backbone = backbone or 'resnet50'
        return UWSegFormer(backbone=backbone, num_classes=num_classes, pretrained=True)
    # elif name == "uwsegformer_v2":
    #     # Enhanced UWSegFormer with color restoration, multi-head attention (deep supervision disabled for stability)
    #     backbone = backbone or 'resnet50'
    #     return UWSegFormerV2(backbone=backbone, num_classes=num_classes, pretrained=True, deep_supervision=False)
    else:
        raise ValueError(f"Unknown model: {name}")

def train_keras_suimnet(args):
    """Train Keras SUIM-Net using the paper's implementation."""
    try:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings
        import tensorflow as tf
        from keras import callbacks
        from models.paper.suimnet import SUIM_Net
        from data.utils.keras_data_utils import trainDataGenerator
    except ImportError as e:
        print("\nError: TensorFlow/Keras not installed.")
        print(f"Details: {e}")
        print("\nPlease install with: pip install tensorflow==2.10.0 keras==2.10.0")
        return
    
    # Setup paths
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    
    # Determine backbone and resolution from args
    if args.backbone and args.backbone.upper() == 'RSB':
        base = 'RSB'
        im_res = (320, 240, 3)
    else:
        base = 'VGG'  # Default
        im_res = (320, 256, 3)
    
    # Map merged classes: 8 classes -> 5 classes for paper compatibility
    n_classes = 5 if args.merge_classes else args.num_classes
    
    class_mode = f"{n_classes}cls"
    aug_mode = "aug" if args.augment else "noaug"
    ckpt_name = f"suimnet_keras_{base.lower()}_{class_mode}_{aug_mode}_best.weights.h5"
    model_ckpt_path = os.path.join(args.ckpt_dir, ckpt_name)
    
    print("=" * 70)
    print("Training Keras SUIM-Net (Paper Implementation)")
    print("=" * 70)
    print(f"Backbone: {base}")
    print(f"Image resolution: {im_res}")
    print(f"Number of classes: {n_classes}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Data augmentation: {'enabled' if args.augment else 'disabled'}")
    print(f"Training data: {args.train_dir}")
    print(f"Checkpoint: {model_ckpt_path}")
    print("=" * 70)
    
    # Initialize SUIM-Net
    print(f"\nInitializing SUIM-Net ({base} backbone)...")
    suimnet = SUIM_Net(base=base, im_res=im_res, n_classes=n_classes)
    model = suimnet.model
    
    # Count parameters
    trainable_count = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    print(f"Parameters: {trainable_count:,}")
    
    # Load checkpoint if resuming
    if args.resume:
        if os.path.exists(args.resume):
            print(f"\nLoading weights from: {args.resume}")
            model.load_weights(args.resume)
        else:
            print(f"\nWarning: Checkpoint {args.resume} not found. Starting fresh.")
    
    # Data augmentation parameters
    if args.augment:
        data_gen_args = dict(
            rotation_range=0.2,
            width_shift_range=0.05,
            height_shift_range=0.05,
            shear_range=0.05,
            zoom_range=0.05,
            horizontal_flip=True,
            fill_mode='nearest'
        )
    else:
        data_gen_args = dict()
    
    # Setup callbacks
    model_checkpoint = callbacks.ModelCheckpoint(
        model_ckpt_path,
        monitor='loss',
        verbose=1,
        mode='auto',
        save_weights_only=True,
        save_best_only=True
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    callback_list = [model_checkpoint, reduce_lr]
    
    print("\nStarting training...")
    print("=" * 70)
    
    # Create data generator
    train_gen = trainDataGenerator(
        batch_size=args.batch_size,
        train_path=args.train_dir,
        image_folder="images",
        mask_folder="masks",
        aug_dict=data_gen_args,
        image_color_mode="rgb",
        mask_color_mode="rgb",
        target_size=(im_res[1], im_res[0]),
        n_classes=n_classes
    )
    
    # Calculate steps per epoch (estimate: 1500 images in SUIM train set)
    steps_per_epoch = max(100, 1500 // args.batch_size)
    print(f"Steps per epoch: {steps_per_epoch}")
    
    # Train model
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=args.epochs,
        callbacks=callback_list,
        verbose=1
    )
    
    print("=" * 70)
    print("Training complete!")
    print(f"Best model saved to: {model_ckpt_path}")
    
    # Save final model
    final_path = model_ckpt_path.replace('_best.hdf5', '_final.hdf5')
    model.save_weights(final_path)
    print(f"Final model saved to: {final_path}")
    print("=" * 70)

def main(args):
    # Check if using Keras model
    if args.model == "suimnet_keras":
        train_keras_suimnet(args)
        return
    
    # Device (supports CUDA, MPS, and CPU)
    device = get_device()
    
    # Datasets
    print(f"\nLoading datasets...")
    print(f"Data augmentation: {'enabled' if args.augment else 'disabled'}")
    print(f"Class mode: {'5 classes (merged, ignoring background)' if args.merge_classes else '8 classes (original)'}")
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
    
    # Track backbone for SUIMNet to include in checkpoint name
    suimnet_backbone = None
    if args.model == "suimnet":
        suimnet_backbone = 'vgg' if args.backbone and args.backbone.upper() == 'VGG' else 'rsb'
        print(f"SUIMNet backbone: {suimnet_backbone.upper()}")
    
    # Loss & Optimizer
    # Use specialized loss for models with deep supervision
    if args.model == "unet_resattn_v4":
        # Class weights for SUIM - boost hard classes (Diver, Plant)
        # Format: [Background, Human_diver, Aquatic_plants, Wreck, Robot, Reefs_invertebrates, Sea_floor_rocks, Fish_vertebrates]
        class_weights = [0.1, 2.5, 3.0, 1.5, 1.5, 1.0, 1.2, 1.0] if args.num_classes == 8 else None
        criterion = V4DeepSupervisionLoss(aux_weight=0.4, edge_weight=0.1, alpha=class_weights, gamma=2.0)
        print("Using V4DeepSupervisionLoss with class weights and deep supervision")
    elif args.model == "suimnet":
        # Binary cross entropy for SUIMNet (matches paper's implementation with sigmoid)
        criterion = BinaryCrossEntropyLoss()
        print("Using BinaryCrossEntropyLoss (matching paper's implementation with sigmoid)")
    else:
        # Standard DiceCE loss for all other models
        criterion = DiceCELoss(dice_weight=0.5)
        # if args.model == "uwsegformer_v2":
        #     print("Using standard DiceCELoss (deep supervision disabled for stability)")
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
    
    # Initialize training plotter
    class_mode = "5cls" if args.merge_classes else "8cls"
    aug_mode = "aug" if args.augment else "noaug"
    plot_name = f"{args.model}_{class_mode}_{aug_mode}_training"
    plotter = TrainingPlotter(save_dir="visualizations", plot_name=plot_name)

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
        
        # Update plotter
        # plotter.update(epoch, train_loss, val_loss, train_iou, val_iou)  # Disabled
        
        # Log
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} mIoU: {train_iou:.4f} | "
              f"Val Loss: {val_loss:.4f} mIoU: {val_iou:.4f}")
        
        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            class_mode = "5cls" if args.merge_classes else "8cls"
            aug_mode = "aug" if args.augment else "noaug"
            save_path = f"checkpoints/{model_name}_{class_mode}_{aug_mode}_best.pth"
            save_checkpoint(model, optimizer, epoch, best_iou, save_path)
            print(f" ★ Saved best model: {save_path} (mIoU: {best_iou:.4f})")
    
    print("=" * 70)
    print(f"Training complete! Best val mIoU: {best_iou:.4f}")
    
    # Save training plots and history
    # print("\nGenerating training plots...")
    # plotter.plot(show=False, save=True)
    # plotter.save_history()
    
    # Print best metrics summary
    # best_metrics = plotter.get_best_metrics()
    print("\nTraining Summary:")
    print(f"  Best Val mIoU: {best_iou:.4f}")
    # print(f"  Min Val Loss: {best_metrics['min_val_loss']:.4f} (Epoch {best_metrics['min_val_loss_epoch']})")
    # print(f"  Final Val mIoU: {best_metrics['final_val_iou']:.4f}")
    
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
    parser.add_argument("--train_dir", default="data",
                       help="Training directory for Keras models (should contain images/ and masks/)")
    parser.add_argument("--ckpt_dir", default="checkpoints",
                       help="Checkpoint directory")
    
    # Model
    parser.add_argument("--model", choices=["unet_resattn", "unet_resattn_v2", "unet_resattn_v3", "unet_resattn_v4",
                                           "suimnet", "suimnet_keras", "deeplabv3", "uwsegformer"],  # "uwsegformer_v2" not implemented
                       default="unet_resattn", help="Model architecture")
    parser.add_argument("--backbone", type=str, default=None,
                       help="Backbone for models. "
                            "PyTorch SUIM-Net: RSB (default) or VGG | "
                            "Keras SUIM-Net: VGG (default) or RSB | "
                            "UWSegFormer: resnet18/34/50/101 (default: resnet50) | "
                            "MiT backbones: mit_b0 (always available), mit_b1/b2/b3/b4/b5 (if installed)")
    parser.add_argument("--merge-classes", action="store_true", default=False,
                       help="Ignore background, plant, and sea_floor_rock (5 classes instead of 8)")
    parser.add_argument("--num_classes", type=int, default=None, 
                       help="Number of classes (auto-set to 5 if --merge-classes, else 8)")
    
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
        args.num_classes = 5 if args.merge_classes else 8
    
    main(args)
