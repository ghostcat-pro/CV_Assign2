"""
Evaluate a trained model on test set.
Run with: python evaluate.py --model unet_resattn --checkpoint checkpoints/unet_resattn_aug_best.pth
"""
import argparse
import torch
from torch.utils.data import DataLoader

from models.unet_resattn import UNetResAttn
from models.suimnet import SUIMNet
from models.deeplab_resnet import get_deeplabv3
from datasets.suim_dataset import SUIMDataset, CLASS_NAMES, CLASS_NAMES_MERGED
from datasets.augmentations import val_transforms
from training.eval import evaluate_loader
from training.utils import load_checkpoint, count_parameters

def get_model(name, num_classes=8):
    """Load model by name."""
    if name == "unet_resattn":
        return UNetResAttn(in_ch=3, out_ch=num_classes, base_ch=64)
    elif name == "suimnet":
        return SUIMNet(in_ch=3, out_ch=num_classes, base=32)
    elif name == "deeplabv3":
        return get_deeplabv3(num_classes=num_classes, pretrained=False)
    else:
        raise ValueError(f"Unknown model: {name}")

def main(args):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test dataset
    print(f"\nLoading test dataset...")
    print(f"Class mode: {'6 classes (merged)' if args.merge_classes else '8 classes (original)'}")
    test_dataset = SUIMDataset(
        split_file=args.test_split,
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        transform=val_transforms,
        merge_classes=args.merge_classes
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                             shuffle=False, num_workers=args.num_workers)
    print(f"Test set: {len(test_dataset)} images")
    
    # Model
    print(f"\nLoading {args.model}...")
    model = get_model(args.model, num_classes=args.num_classes)
    
    # Load checkpoint
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        load_checkpoint(model, None, args.checkpoint, device)
    else:
        print("WARNING: No checkpoint provided, using random weights!")
    
    model = model.to(device)
    print(f"Parameters: {count_parameters(model):,}")
    
    # Evaluate
    print("\n" + "=" * 70)
    print("Evaluating on test set...")
    print("=" * 70)
    
    miou, per_class = evaluate_loader(model, test_loader, device, args.num_classes)
    
    print(f"\nmean IoU: {miou:.4f}")
    print("\nPer-class IoU:")
    print("-" * 50)
    class_names = CLASS_NAMES_MERGED if args.merge_classes else CLASS_NAMES
    for name, iou in zip(class_names, per_class):
        print(f"{name:25s}: {iou:.4f}")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SUIM segmentation model")
    
    # Data
    parser.add_argument("--test_split", default="data/test.txt")
    parser.add_argument("--images_dir", default="data/images")
    parser.add_argument("--masks_dir", default="data/masks")
    
    # Model
    parser.add_argument("--model", choices=["unet_resattn", "suimnet", "deeplabv3"],
                       required=True, help="Model architecture")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--merge-classes", action="store_true", default=False,
                       help="Merge background, plant, and sea_floor_rock into one class (6 classes)")
    parser.add_argument("--num_classes", type=int, default=None,
                       help="Number of classes (auto-set to 6 if --merge-classes, else 8)")
    
    # Eval
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    
    args = parser.parse_args()
    
    # Auto-set num_classes if not explicitly provided
    if args.num_classes is None:
        args.num_classes = 6 if args.merge_classes else 8
    
    main(args)
