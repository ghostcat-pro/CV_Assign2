"""
Evaluate all trained models with IoU and F-score metrics
No retraining needed - uses existing checkpoints
"""

import torch
from torch.utils.data import DataLoader
import argparse

from models.suimnet import SUIMNet
from models.unet_resattn import UNetResAttn
from models.unet_resattn_v2 import UNetResAttnV2
from models.unet_resattn_v3 import UNetResAttnV3
from models.deeplab_resnet import get_deeplabv3
from datasets.suim_dataset import SUIMDataset, CLASS_NAMES, CLASS_NAMES_MERGED
from datasets.augmentations import val_transforms
from training.metrics import evaluate_model_full


CLASS_NAMES_DISPLAY = [
    "Background",
    "Diver", 
    "Plant",
    "Wreck",
    "Robot",
    "Reef/Invertebrate",
    "Fish/Vertebrate",
    "Sea-floor/Rock"
]

CLASS_NAMES_MERGED_DISPLAY = [
    "Background/Plant/Seafloor",
    "Diver",
    "Wreck",
    "Robot",
    "Reef/Invertebrate",
    "Fish/Vertebrate"
]


def load_model(model_name, checkpoint_path, device, num_classes=8):
    """Load model and checkpoint"""
    
    # Initialize model
    if model_name == 'suimnet':
        model = SUIMNet(in_ch=3, out_ch=num_classes)
    elif model_name == 'unet_resattn':
        model = UNetResAttn(in_ch=3, out_ch=num_classes)
    elif model_name == 'unet_resattn_v2':
        model = UNetResAttnV2(in_ch=3, out_ch=num_classes, deep_supervision=True)
    elif model_name == 'unet_resattn_v3':
        model = UNetResAttnV3(in_ch=3, out_ch=num_classes, pretrained=False)
    elif model_name == 'deeplabv3':
        model = get_deeplabv3(num_classes=num_classes, pretrained=False)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Handle DeepLabV3 aux_classifier mismatch
    if model_name == 'deeplabv3':
        state_dict = {k: v for k, v in state_dict.items() if 'aux_classifier' not in k}
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()
    
    return model


def evaluate_all_models(test_file='data/test.txt', batch_size=8, merge_classes=False):
    """Evaluate all trained models with IoU and F-score"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    num_classes = 6 if merge_classes else 8
    class_names_display = CLASS_NAMES_MERGED_DISPLAY if merge_classes else CLASS_NAMES_DISPLAY
    print(f"Class mode: {'6 classes (merged)' if merge_classes else '8 classes (original)'}\n")
    
    # Load test dataset
    test_dataset = SUIMDataset(test_file, transform=val_transforms, merge_classes=merge_classes)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=4)
    print(f"Test set: {len(test_dataset)} images\n")
    
    # Models to evaluate
    models_to_eval = [
        {
            'name': 'SUIM-Net',
            'model_name': 'suimnet',
            'checkpoint': 'checkpoints/suimnet_aug_best.pth',
            'resolution': 256
        },
        {
            'name': 'UNet-ResAttn',
            'model_name': 'unet_resattn',
            'checkpoint': 'checkpoints/unet_resattn_aug_best.pth',
            'resolution': 256
        },
        {
            'name': 'UNet-ResAttn-V2',
            'model_name': 'unet_resattn_v2',
            'checkpoint': 'checkpoints/unet_resattn_v2_best.pth',
            'resolution': 256
        },
        {
            'name': 'UNet-ResAttn-V3',
            'model_name': 'unet_resattn_v3',
            'checkpoint': 'checkpoints/unet_resattn_v3_best.pth',
            'resolution': 384
        },
        {
            'name': 'DeepLabV3-ResNet50',
            'model_name': 'deeplabv3',
            'checkpoint': 'checkpoints/deeplabv3_aug_best.pth',
            'resolution': 256
        }
    ]
    
    results = []
    
    print("=" * 80)
    print("EVALUATING ALL MODELS WITH IoU AND F-SCORE METRICS")
    print("=" * 80)
    print()
    
    for model_info in models_to_eval:
        print(f"Evaluating {model_info['name']}...")
        print(f"Checkpoint: {model_info['checkpoint']}")
        
        try:
            # Load model
            model = load_model(
                model_info['model_name'],
                model_info['checkpoint'],
                device,
                num_classes=num_classes
            )
            
            # Evaluate with both metrics
            metrics = evaluate_model_full(model, test_loader, device, num_classes=num_classes)
            
            results.append({
                'name': model_info['name'],
                'metrics': metrics
            })
            
            print(f"✓ Mean IoU: {metrics['mean_iou']*100:.2f}%")
            print(f"✓ Mean F-score: {metrics['mean_fscore']*100:.2f}%")
            print()
            
        except FileNotFoundError:
            print(f"✗ Checkpoint not found: {model_info['checkpoint']}")
            print()
        except Exception as e:
            print(f"✗ Error: {e}")
            print()
    
    # Print comprehensive results table
    print("=" * 80)
    print("SUMMARY RESULTS")
    print("=" * 80)
    print()
    
    # Overall metrics table
    print("Overall Performance:")
    print("-" * 60)
    print(f"{'Model':<25} {'Mean IoU':<15} {'Mean F-score':<15}")
    print("-" * 60)
    for result in results:
        name = result['name']
        iou = result['metrics']['mean_iou'] * 100
        fscore = result['metrics']['mean_fscore'] * 100
        print(f"{name:<25} {iou:>6.2f}%         {fscore:>6.2f}%")
    print("-" * 60)
    print()
    
    # Per-class IoU table
    print("Per-Class IoU (%):")
    print("-" * 100)
    header = f"{'Class':<20}"
    for result in results:
        header += f"{result['name']:<20}"
    print(header)
    print("-" * 100)
    
    for i, class_name in enumerate(CLASS_NAMES):
        row = f"{class_name:<20}"
        for result in results:
            iou = result['metrics']['iou_per_class'][i]
            if np.isnan(iou):
                row += f"{'N/A':<20}"
            else:
                row += f"{iou*100:>6.2f}%{' '*13}"
        print(row)
    print("-" * 100)
    print()
    
    # Per-class F-score table
    print("Per-Class F-score (%):")
    print("-" * 100)
    header = f"{'Class':<20}"
    for result in results:
        header += f"{result['name']:<20}"
    print(header)
    print("-" * 100)
    
    for i, class_name in enumerate(CLASS_NAMES):
        row = f"{class_name:<20}"
        for result in results:
            fscore = result['metrics']['fscore_per_class'][i]
            if np.isnan(fscore):
                row += f"{'N/A':<20}"
            else:
                row += f"{fscore*100:>6.2f}%{' '*13}"
        print(row)
    print("-" * 100)
    print()
    
    # Save results to file
    with open('evaluation_results_with_fscore.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("COMPREHENSIVE EVALUATION RESULTS - IoU AND F-SCORE\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Overall Performance:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Model':<25} {'Mean IoU':<15} {'Mean F-score':<15}\n")
        f.write("-" * 60 + "\n")
        for result in results:
            name = result['name']
            iou = result['metrics']['mean_iou'] * 100
            fscore = result['metrics']['mean_fscore'] * 100
            f.write(f"{name:<25} {iou:>6.2f}%         {fscore:>6.2f}%\n")
        f.write("-" * 60 + "\n\n")
        
        f.write("\n\nPer-Class IoU (%):\n")
        for i, class_name in enumerate(class_names_display):
            f.write(f"\n{class_name}:\n")
            for result in results:
                iou = result['metrics']['iou_per_class'][i]
                if not np.isnan(iou):
                    f.write(f"  {result['name']:<25} {iou*100:>6.2f}%\n")
        
        f.write("\n\nPer-Class F-score (%):\n")
        for i, class_name in enumerate(class_names_display):
            f.write(f"\n{class_name}:\n")
            for result in results:
                fscore = result['metrics']['fscore_per_class'][i]
                if not np.isnan(fscore):
                    f.write(f"  {result['name']:<25} {fscore*100:>6.2f}%\n")
    
    print("Results saved to: evaluation_results_with_fscore.txt")
    print()


if __name__ == "__main__":
    import numpy as np
    
    parser = argparse.ArgumentParser(description='Evaluate all models with IoU and F-score')
    parser.add_argument('--test_file', type=str, default='data/test.txt',
                       help='Path to test split file')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for evaluation')
    parser.add_argument('--merge-classes', action='store_true', default=False,
                       help='Merge background, plant, and sea_floor_rock into one class (6 classes)')
    
    args = parser.parse_args()
    
    evaluate_all_models(args.test_file, args.batch_size, args.merge_classes)
