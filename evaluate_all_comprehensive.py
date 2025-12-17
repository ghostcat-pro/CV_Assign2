"""
Comprehensive evaluation of ALL models including:
- PyTorch models with augmentation and no-augmentation variants
- Keras SUIM-Net models (VGG and RSB backbones)
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force Keras/TensorFlow to use CPU only

from models.suimnet import SUIMNet
from models.unet_resattn import UNetResAttn
from models.unet_resattn_v2 import UNetResAttnV2
from models.unet_resattn_v3 import UNetResAttnV3
from models.unet_resattn_v4 import UNetResAttnV4
from models.deeplab_resnet import get_deeplabv3
from models.uwsegformer import UWSegFormer
from datasets.suim_dataset import SUIMDataset
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


def load_pytorch_model(model_name, checkpoint_path, device, num_classes=8):
    """Load PyTorch model and checkpoint"""
    
    # Initialize model
    if model_name == 'suimnet':
        model = SUIMNet(in_ch=3, out_ch=num_classes)
    elif model_name == 'unet_resattn':
        model = UNetResAttn(in_ch=3, out_ch=num_classes)
    elif model_name == 'unet_resattn_v2':
        model = UNetResAttnV2(in_ch=3, out_ch=num_classes, deep_supervision=True)
    elif model_name == 'unet_resattn_v3':
        model = UNetResAttnV3(in_ch=3, out_ch=num_classes, pretrained=False)
    elif model_name == 'unet_resattn_v4':
        model = UNetResAttnV4(in_ch=3, out_ch=num_classes, pretrained=False, deep_supervision=True)
    elif model_name == 'deeplabv3':
        model = get_deeplabv3(num_classes=num_classes, pretrained=False)
    elif model_name == 'uwsegformer':
        model = UWSegFormer(backbone='resnet50', num_classes=num_classes, pretrained=False)
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


def load_keras_model(backbone, checkpoint_path, num_classes=8):
    """Load Keras SUIM-Net model"""
    import tensorflow as tf
    from models.paper.suimnet import SUIM_Net
    
    # Build model - SUIM_Net expects (H, W, C) format
    suimnet = SUIM_Net(base=backbone, im_res=(320, 256, 3), n_classes=num_classes)
    model = suimnet.model
    
    # Load weights
    model.load_weights(checkpoint_path)
    
    return model


def evaluate_keras_model(model, test_loader, num_classes=8, target_size=(256, 320)):
    """Evaluate Keras model with IoU and F-score"""
    import tensorflow as tf
    
    all_ious = []
    all_fscores = []
    class_ious = np.zeros(num_classes)
    class_fscores = np.zeros(num_classes)
    class_counts = np.zeros(num_classes)
    
    for images, masks in test_loader:
        # Convert to numpy and transpose to NHWC format
        images_np = images.numpy().transpose(0, 2, 3, 1)  # NCHW -> NHWC
        masks_np = masks.numpy()
        
        # Resize to expected input size (H, W)
        import cv2
        batch_size = images_np.shape[0]
        resized_images = np.zeros((batch_size, target_size[0], target_size[1], 3), dtype=np.float32)
        resized_masks = np.zeros((batch_size, target_size[0], target_size[1]), dtype=np.int64)
        
        for i in range(batch_size):
            resized_images[i] = cv2.resize(images_np[i], (target_size[1], target_size[0]), 
                                          interpolation=cv2.INTER_LINEAR)
            resized_masks[i] = cv2.resize(masks_np[i].astype(np.uint8), (target_size[1], target_size[0]), 
                                         interpolation=cv2.INTER_NEAREST)
        
        # Predict
        preds = model.predict(resized_images, verbose=0)
        preds = np.argmax(preds, axis=-1)
        
        # Calculate metrics for each image
        for pred, mask in zip(preds, resized_masks):
            # IoU per class
            for cls in range(num_classes):
                pred_mask = (pred == cls)
                true_mask = (mask == cls)
                
                intersection = np.logical_and(pred_mask, true_mask).sum()
                union = np.logical_or(pred_mask, true_mask).sum()
                
                if union > 0:
                    iou = intersection / union
                    class_ious[cls] += iou
                    class_counts[cls] += 1
                    
                    # F-score (Dice)
                    if (pred_mask.sum() + true_mask.sum()) > 0:
                        fscore = 2 * intersection / (pred_mask.sum() + true_mask.sum())
                        class_fscores[cls] += fscore
    
    # Average per class
    class_ious = np.divide(class_ious, class_counts, where=class_counts > 0)
    class_fscores = np.divide(class_fscores, class_counts, where=class_counts > 0)
    
    # Handle NaN values
    class_ious = np.nan_to_num(class_ious, nan=0.0)
    class_fscores = np.nan_to_num(class_fscores, nan=0.0)
    
    return {
        'mean_iou': np.mean(class_ious),
        'mean_fscore': np.mean(class_fscores),
        'iou_per_class': class_ious,
        'fscore_per_class': class_fscores
    }


def evaluate_all_comprehensive(test_file='data/test.txt', batch_size=8):
    """Evaluate ALL models comprehensively"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    num_classes = 8
    
    # Load test dataset
    test_dataset = SUIMDataset(test_file, transform=val_transforms, merge_classes=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=4)
    print(f"Test set: {len(test_dataset)} images\n")
    
    # All models to evaluate
    models_to_eval = [
        # PyTorch models - Augmented
        {
            'name': 'SUIM-Net (PyTorch, Aug)',
            'type': 'pytorch',
            'model_name': 'suimnet',
            'checkpoint': 'checkpoints/suimnet_8cls_aug_best.pth',
            'params': 7.76
        },
        {
            'name': 'UNet-ResAttn (Aug)',
            'type': 'pytorch',
            'model_name': 'unet_resattn',
            'checkpoint': 'checkpoints/unet_resattn_8cls_aug_best.pth',
            'params': 32.96
        },
        {
            'name': 'UNet-ResAttn-V3 (Aug)',
            'type': 'pytorch',
            'model_name': 'unet_resattn_v3',
            'checkpoint': 'checkpoints/unet_resattn_v3_8cls_aug_best.pth',
            'params': 74.49
        },
        {
            'name': 'UNet-ResAttn-V3 (No Aug)',
            'type': 'pytorch',
            'model_name': 'unet_resattn_v3',
            'checkpoint': 'checkpoints/unet_resattn_v3_8cls_noaug_best.pth',
            'params': 74.49
        },
        {
            'name': 'UNet-ResAttn-V4 (Aug)',
            'type': 'pytorch',
            'model_name': 'unet_resattn_v4',
            'checkpoint': 'checkpoints/unet_resattn_v4_8cls_aug_best.pth',
            'params': 138.15
        },
        {
            'name': 'UNet-ResAttn-V4 (No Aug)',
            'type': 'pytorch',
            'model_name': 'unet_resattn_v4',
            'checkpoint': 'checkpoints/unet_resattn_v4_8cls_noaug_best.pth',
            'params': 138.15
        },
        {
            'name': 'DeepLabV3-ResNet50 (Aug)',
            'type': 'pytorch',
            'model_name': 'deeplabv3',
            'checkpoint': 'checkpoints/deeplabv3_8cls_aug_best.pth',
            'params': 42.00
        },
        {
            'name': 'DeepLabV3-ResNet50 (No Aug)',
            'type': 'pytorch',
            'model_name': 'deeplabv3',
            'checkpoint': 'checkpoints/deeplabv3_8cls_noaug_best.pth',
            'params': 42.00
        },
        {
            'name': 'UWSegFormer (Aug)',
            'type': 'pytorch',
            'model_name': 'uwsegformer',
            'checkpoint': 'checkpoints/uwsegformer_8cls_aug_best.pth',
            'params': 30.24
        },
        {
            'name': 'UWSegFormer (No Aug)',
            'type': 'pytorch',
            'model_name': 'uwsegformer',
            'checkpoint': 'checkpoints/uwsegformer_8cls_noaug_best.pth',
            'params': 30.24
        },
        # Keras models
        {
            'name': 'SUIM-Net Keras VGG (Aug)',
            'type': 'keras',
            'backbone': 'VGG',
            'checkpoint': 'checkpoints/suimnet_keras_vgg_8cls_aug_best.weights.h5',
            'params': 33.64
        },
        {
            'name': 'SUIM-Net Keras VGG (No Aug)',
            'type': 'keras',
            'backbone': 'VGG',
            'checkpoint': 'checkpoints/suimnet_keras_vgg_8cls_noaug_best.weights.h5',
            'params': 33.64
        },
        {
            'name': 'SUIM-Net Keras RSB (Aug)',
            'type': 'keras',
            'backbone': 'RSB',
            'checkpoint': 'checkpoints/suimnet_keras_rsb_8cls_aug_best.weights.h5',
            'params': 11.20
        },
        {
            'name': 'SUIM-Net Keras RSB (No Aug)',
            'type': 'keras',
            'backbone': 'RSB',
            'checkpoint': 'checkpoints/suimnet_keras_rsb_8cls_noaug_best.weights.h5',
            'params': 11.20
        },
        # V2 (no aug only)
        {
            'name': 'UNet-ResAttn-V2 (No Aug)',
            'type': 'pytorch',
            'model_name': 'unet_resattn_v2',
            'checkpoint': 'checkpoints/unet_resattn_v2_8cls_noaug_best.pth',
            'params': 68.86
        }
    ]
    
    results = []
    
    print("=" * 100)
    print("COMPREHENSIVE EVALUATION - ALL MODELS (PYTORCH + KERAS, AUG + NO-AUG)")
    print("=" * 100)
    print()
    
    for model_info in models_to_eval:
        print(f"Evaluating {model_info['name']}...")
        print(f"Checkpoint: {model_info['checkpoint']}")
        
        if not os.path.exists(model_info['checkpoint']):
            print(f"✗ Checkpoint not found, skipping...")
            print()
            continue
        
        try:
            if model_info['type'] == 'pytorch':
                # Load PyTorch model
                model = load_pytorch_model(
                    model_info['model_name'],
                    model_info['checkpoint'],
                    device,
                    num_classes=num_classes
                )
                
                # Evaluate
                metrics = evaluate_model_full(model, test_loader, device, num_classes=num_classes)
                
            elif model_info['type'] == 'keras':
                # Load Keras model
                model = load_keras_model(
                    model_info['backbone'],
                    model_info['checkpoint'],
                    num_classes=num_classes
                )
                
                # Evaluate
                metrics = evaluate_keras_model(model, test_loader, num_classes=num_classes)
            
            results.append({
                'name': model_info['name'],
                'metrics': metrics,
                'params': model_info['params']
            })
            
            print(f"✓ Mean IoU: {metrics['mean_iou']*100:.2f}%")
            print(f"✓ Mean F-score: {metrics['mean_fscore']*100:.2f}%")
            print()
            
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    # Save comprehensive results
    save_comprehensive_results(results, CLASS_NAMES_DISPLAY)
    
    return results


def save_comprehensive_results(results, class_names):
    """Save comprehensive results to file"""
    
    output_file = 'evaluation_comprehensive_results.txt'
    
    with open(output_file, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("COMPREHENSIVE EVALUATION RESULTS - ALL MODELS\n")
        f.write("PyTorch + Keras | Augmentation + No-Augmentation Variants\n")
        f.write("=" * 100 + "\n\n")
        
        # Overall performance table
        f.write("Overall Performance:\n")
        f.write("-" * 90 + "\n")
        f.write(f"{'Model':<40} {'Params (M)':<12} {'Mean IoU':<12} {'Mean F-score':<12}\n")
        f.write("-" * 90 + "\n")
        
        # Sort by IoU
        sorted_results = sorted(results, key=lambda x: x['metrics']['mean_iou'], reverse=True)
        
        for result in sorted_results:
            name = result['name']
            params = result['params']
            iou = result['metrics']['mean_iou'] * 100
            fscore = result['metrics']['mean_fscore'] * 100
            f.write(f"{name:<40} {params:>6.2f}M      {iou:>6.2f}%      {fscore:>6.2f}%\n")
        f.write("-" * 90 + "\n\n")
        
        # Per-class IoU
        f.write("\nPer-Class IoU (%):\n")
        f.write("=" * 100 + "\n")
        for i, class_name in enumerate(class_names):
            f.write(f"\n{class_name}:\n")
            f.write("-" * 70 + "\n")
            for result in sorted_results:
                iou = result['metrics']['iou_per_class'][i]
                if not np.isnan(iou) and iou > 0:
                    f.write(f"  {result['name']:<40} {iou*100:>6.2f}%\n")
        
        # Per-class F-score
        f.write("\n\nPer-Class F-score (%):\n")
        f.write("=" * 100 + "\n")
        for i, class_name in enumerate(class_names):
            f.write(f"\n{class_name}:\n")
            f.write("-" * 70 + "\n")
            for result in sorted_results:
                fscore = result['metrics']['fscore_per_class'][i]
                if not np.isnan(fscore) and fscore > 0:
                    f.write(f"  {result['name']:<40} {fscore*100:>6.2f}%\n")
    
    print(f"\nResults saved to: {output_file}")
    print()


if __name__ == "__main__":
    evaluate_all_comprehensive()
