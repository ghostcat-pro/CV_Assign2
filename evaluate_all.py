#!/usr/bin/env python3
"""
Evaluate all trained models on the test set.
Usage: python evaluate_all.py
"""

import torch
import argparse
from pathlib import Path
from datasets.suim_dataset import SUIMDataset
from datasets.augmentations import val_transforms
from torch.utils.data import DataLoader
from training.eval import evaluate_loader
from models.unet_resattn import UNetResAttn
from models.suimnet import SUIMNet
from models.deeplab_resnet import get_deeplabv3

def get_model(model_name, num_classes=8):
    """Initialize the model."""
    if model_name == "unet_resattn":
        return UNetResAttn(in_ch=3, out_ch=num_classes)
    elif model_name == "suimnet":
        return SUIMNet(in_ch=3, out_ch=num_classes)
    elif model_name == "deeplabv3":
        return get_deeplabv3(num_classes=num_classes, pretrained=False)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate_model(model_name, checkpoint_path, test_loader, device, num_classes=8):
    """Evaluate a single model."""
    print(f"\n{'='*70}")
    print(f"Evaluating: {model_name.upper()}")
    print(f"{'='*70}")
    
    # Load model
    model = get_model(model_name, num_classes)
    model = model.to(device)
    print(f"Parameters: {count_parameters(model):,}")
    
    # Load checkpoint
    if Path(checkpoint_path).exists():
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        
        # Handle DeepLabV3 aux_classifier mismatch
        if model_name == "deeplabv3":
            state_dict = ckpt["model"]
            # Remove aux_classifier keys if they exist
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("aux_classifier")}
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(ckpt["model"])
        
        if "epoch" in ckpt:
            print(f"Trained for {ckpt['epoch']} epochs")
        if "best_iou" in ckpt:
            print(f"Best validation mIoU: {ckpt['best_iou']:.4f}")
    else:
        print(f"Warning: Checkpoint not found: {checkpoint_path}")
        return None
    
    # Evaluate
    print(f"\nEvaluating on test set...")
    model.eval()
    mean_iou, per_class_iou = evaluate_loader(model, test_loader, device, num_classes)
    
    print(f"\nTest mIoU: {mean_iou:.4f} ({mean_iou*100:.2f}%)")
    print(f"\nPer-class IoU:")
    print("-" * 50)
    
    class_names = [
        "background", "diver", "plant", "wreck",
        "robot", "reef_invertebrate", "fish_vertebrate", "sea_floor_rock"
    ]
    
    valid_classes = []
    for i, (name, iou) in enumerate(zip(class_names, per_class_iou)):
        if not torch.isnan(torch.tensor(iou)):
            valid_classes.append((name, iou))
            print(f"{name:20s} : {iou:.4f} ({iou*100:.2f}%)")
        else:
            print(f"{name:20s} : N/A (not in test set)")
    
    return {
        "model": model_name,
        "mean_iou": mean_iou,
        "per_class_iou": per_class_iou,
        "parameters": count_parameters(model),
        "valid_classes": valid_classes
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load test dataset
    print("Loading test dataset...")
    test_dataset = SUIMDataset(
        split_file="data/test.txt",
        images_dir="data/images",
        masks_dir="data/masks",
        transform=val_transforms
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    print(f"Test set: {len(test_dataset)} images\n")
    
    # Models to evaluate
    models = [
        ("suimnet", "checkpoints/suimnet_aug_best.pth"),
        ("unet_resattn", "checkpoints/unet_resattn_aug_best.pth"),
        ("deeplabv3", "checkpoints/deeplabv3_aug_best.pth"),
    ]
    
    results = []
    for model_name, checkpoint_path in models:
        result = evaluate_model(model_name, checkpoint_path, test_loader, device)
        if result:
            results.append(result)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY - All Models on Test Set")
    print(f"{'='*70}")
    print(f"{'Model':<20} {'Parameters':<15} {'Test mIoU':<12}")
    print("-" * 70)
    
    for res in results:
        print(f"{res['model']:<20} {res['parameters']:>12,}   {res['mean_iou']:>8.4f} ({res['mean_iou']*100:>5.2f}%)")
    
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
