"""
Evaluate all trained models with IoU and F-score metrics
No retraining needed - uses existing checkpoints
"""

import torch
from torch.utils.data import DataLoader
import argparse
import os
import numpy as np

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from models.suimnet import SUIMNet
from models.unet_resattn import UNetResAttn
from models.unet_resattn_v2 import UNetResAttnV2
from models.unet_resattn_v3 import UNetResAttnV3
from models.unet_resattn_v4 import UNetResAttnV4
from models.deeplab_resnet import get_deeplabv3
from models.uwsegformer import UWSegFormer
# from models.uwsegformer_v2 import UWSegFormerV2  # Not implemented yet
from datasets.suim_dataset import SUIMDataset, CLASS_NAMES, CLASS_NAMES_MERGED
from datasets.augmentations import val_transforms
from training.metrics import evaluate_model_full


# 5 classes only (ignoring background, plant, sea-floor)
CLASS_NAMES_5CLS = [
    "Diver",
    "Wreck",
    "Robot",
    "Reef/Invertebrate",
    "Fish/Vertebrate"
]


def load_keras_model(backbone, checkpoint_path, num_classes=5, im_res=(320, 240, 3)):
    """Load Keras SUIM-Net model"""
    try:
        import tensorflow as tf
        from models.paper.suimnet import SUIM_Net
    except ImportError as e:
        raise ImportError(f"TensorFlow/Keras not installed: {e}")
    
    # Initialize model
    suimnet = SUIM_Net(base=backbone, im_res=im_res, n_classes=num_classes)
    model = suimnet.model
    
    # Load weights
    if os.path.exists(checkpoint_path):
        model.load_weights(checkpoint_path)
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    return model


def evaluate_keras_model(model, test_loader, num_classes=5, im_res=(320, 240)):
    """Evaluate Keras model with IoU and boundary-based F-score
    
    SUIM-Net paper uses 5 object classes (NO background channel):
    - Uses multi-label sigmoid output for 5 classes
    - Background/plants/seafloor/rocks are implicitly "not predicted" (no output channel)
    
    Keras SUIM-Net uses multi-label sigmoid output (not softmax):
    - Keras outputs: (B, H, W, 5) with sigmoid probabilities [0, 1] per class
    - Each pixel can belong to multiple classes simultaneously
    
    Keras model class mapping (correct SUIM paper implementation):
    - Keras Class 0: Diver -> PyTorch Class 0
    - Keras Class 1: Wreck -> PyTorch Class 1
    - Keras Class 2: Robot -> PyTorch Class 2
    - Keras Class 3: Reef -> PyTorch Class 3
    - Keras Class 4: Fish -> PyTorch Class 4
    
    PyTorch ground truth classes (5): Diver(0), Wreck(1), Robot(2), Reef(3), Fish(4)
    
    Metrics:
    - IoU: Standard pixel-wise Intersection over Union
    - F-score: Boundary-based F-measure (DAVIS benchmark) matching original Keras evaluation
    """
    import tensorflow as tf
    
    print("SUIM-Net uses 5 object classes (Diver, Wreck, Robot, Reef, Fish)")
    print("Background/plants/seafloor are NOT predicted (no output channel)")
    print("Using boundary-based F-score (DAVIS benchmark) for evaluation.")
    
    all_ious = []
    all_fscores = []
    
    for images, masks in test_loader:
        # Convert to numpy and resize
        images_np = images.numpy().transpose(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        masks_np = masks.numpy()
        
        batch_size = images_np.shape[0]
        resized_images = np.zeros((batch_size, im_res[1], im_res[0], 3), dtype=np.float32)
        
        for i in range(batch_size):
            # Normalize to [0, 1] as expected by Keras model
            img_normalized = images_np[i] / 255.0 if images_np[i].max() > 1.0 else images_np[i]
            resized_images[i] = tf.image.resize(img_normalized, (im_res[1], im_res[0])).numpy()
        
        # Predict (output: (B, H, W, 5) with sigmoid activations in [0, 1])
        preds_prob = model.predict(resized_images, verbose=0)
        
        # Process each image in batch
        for i in range(batch_size):
            # Get sigmoid probabilities for this image: (H, W, 5)
            pred_probs = preds_prob[i]  # Shape: (im_res[1], im_res[0], 5)
            
            # Resize prediction to match ground truth mask size
            pred_probs_resized = tf.image.resize(
                pred_probs[np.newaxis, :, :, :],  # Add batch dim
                (masks_np.shape[1], masks_np.shape[2]),
                method='bilinear'
            ).numpy()[0]  # Remove batch dim -> (H, W, 5)
            
            # Apply threshold to get binary predictions per class
            # Use 0.5 threshold for sigmoid outputs
            pred_binary = (pred_probs_resized > 0.5).astype(np.float32)  # (H, W, 5)
            
            # Map Keras classes to PyTorch classes (now 1:1 mapping!)
            # Keras: [0=Diver, 1=Wreck, 2=Robot, 3=Reef, 4=Fish]
            # PyTorch: [0=Diver, 1=Wreck, 2=Robot, 3=Reef, 4=Fish] (with 255=ignore)
            
            # Since Keras uses multi-label sigmoid, we need to select dominant class per pixel
            # Strategy: Take the class with highest probability
            h, w = pred_probs_resized.shape[:2]
            
            # Get the Keras class with max probability per pixel
            keras_pred = np.argmax(pred_probs_resized, axis=-1)  # (H, W)
            
            # Get max probability value to filter out low-confidence predictions
            max_prob = np.max(pred_probs_resized, axis=-1)  # (H, W)
            
            # Create PyTorch prediction with 1:1 class mapping
            pytorch_pred = keras_pred.copy().astype(np.int64)  # Direct mapping: 0->0, 1->1, 2->2, 3->3, 4->4
            
            # Set low-confidence predictions (< 0.5 threshold) to ignore (255)
            # This handles background/plants/seafloor which have no output channel
            pytorch_pred[max_prob < 0.5] = 255
            
            mask = masks_np[i]
            
            # Calculate metrics per image
            iou = calculate_iou_per_class_numpy(pytorch_pred, mask, 5)
            # Use boundary-based F-score for Keras models (matches original evaluation)
            fscore = calculate_boundary_fscore_per_class_numpy(pytorch_pred, mask, 5)
            
            all_ious.append(iou)
            all_fscores.append(fscore)
    
    # Average across all images
    all_ious = np.array(all_ious)
    all_fscores = np.array(all_fscores)
    
    mean_iou_per_class = np.nanmean(all_ious, axis=0)
    mean_fscore_per_class = np.nanmean(all_fscores, axis=0)
    
    mean_iou = np.nanmean(mean_iou_per_class)
    mean_fscore = np.nanmean(mean_fscore_per_class)
    
    return {
        'mean_iou': mean_iou,
        'iou_per_class': mean_iou_per_class,
        'mean_fscore': mean_fscore,
        'fscore_per_class': mean_fscore_per_class
    }


def calculate_iou_per_class_numpy(pred, target, num_classes):
    """Calculate IoU for each class using numpy"""
    ious = []
    pred = pred.flatten()
    target = target.flatten()
    
    for cls in range(num_classes):
        pred_mask = pred == cls
        target_mask = target == cls
        
        intersection = np.sum(pred_mask & target_mask)
        union = np.sum(pred_mask | target_mask)
        
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    
    return np.array(ious)


def seg2bmap(seg):
    """
    Convert segmentation mask to boundary map.
    From DAVIS benchmark: https://github.com/fperazzi/davis
    
    Arguments:
        seg: Binary segmentation mask (values 0 or 1)
    Returns:
        bmap: Binary boundary map
    """
    seg = seg.astype(bool)
    
    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)
    
    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]
    
    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0
    
    return b


def db_eval_boundary(foreground_mask, gt_mask, bound_th=0.008):
    """
    Compute boundary-based precision, recall, and F-measure.
    From DAVIS benchmark: https://github.com/fperazzi/davis
    
    Arguments:
        foreground_mask: binary segmentation image
        gt_mask: binary annotated image
        bound_th: boundary thickness threshold (default 0.008)
    Returns:
        P: boundaries precision
        R: boundaries recall
        F: boundaries F-measure
    """
    try:
        from skimage.morphology import binary_dilation, disk
    except ImportError:
        raise ImportError("scikit-image required for boundary F-score. Install with: pip install scikit-image")
    
    bound_pix = bound_th if bound_th >= 1 else \
                np.ceil(bound_th * np.linalg.norm(foreground_mask.shape))
    
    # Get the pixel boundaries of both masks
    fg_boundary = seg2bmap(foreground_mask)
    gt_boundary = seg2bmap(gt_mask)
    
    fg_dil = binary_dilation(fg_boundary, disk(bound_pix))
    gt_dil = binary_dilation(gt_boundary, disk(bound_pix))
    
    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil
    
    # Area of the intersection
    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)
    
    # Compute precision and recall
    if n_fg == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)
    
    # Compute F measure
    if precision + recall == 0:
        F = 0
    else:
        F = 2 * precision * recall / (precision + recall)
    
    return precision, recall, F


def calculate_fscore_per_class_numpy(pred, target, num_classes):
    """Calculate F-score for each class using numpy (pixel-based Dice coefficient)"""
    fscores = []
    pred = pred.flatten()
    target = target.flatten()
    
    for cls in range(num_classes):
        pred_mask = pred == cls
        target_mask = target == cls
        
        intersection = np.sum(pred_mask & target_mask)
        pred_area = np.sum(pred_mask)
        target_area = np.sum(target_mask)
        
        if pred_area + target_area == 0:
            fscores.append(float('nan'))
        else:
            fscore = 2 * intersection / (pred_area + target_area)
            fscores.append(fscore)
    
    return np.array(fscores)


def calculate_boundary_fscore_per_class_numpy(pred, target, num_classes):
    """Calculate boundary-based F-score for each class (DAVIS benchmark metric)"""
    fscores = []
    
    for cls in range(num_classes):
        pred_mask = (pred == cls).astype(np.uint8)
        target_mask = (target == cls).astype(np.uint8)
        
        # Skip if class not present in either mask
        if np.sum(pred_mask) == 0 and np.sum(target_mask) == 0:
            fscores.append(float('nan'))
        else:
            try:
                _, _, F = db_eval_boundary(pred_mask, target_mask)
                fscores.append(F)
            except:
                fscores.append(float('nan'))
    
    return np.array(fscores)


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
    elif model_name == 'unet_resattn_v4':
        model = UNetResAttnV4(in_ch=3, out_ch=num_classes, pretrained=False, deep_supervision=True)
    elif model_name == 'deeplabv3':
        model = get_deeplabv3(num_classes=num_classes, pretrained=False)
    elif model_name == 'uwsegformer':
        model = UWSegFormer(backbone='resnet50', num_classes=num_classes, pretrained=False)
    # elif model_name == 'uwsegformer_v2':
    #     model = UWSegFormerV2(backbone='resnet50', num_classes=num_classes, pretrained=False, deep_supervision=True)
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


def evaluate_all_models(test_file='data/test.txt', batch_size=8, merge_classes=True):
    """Evaluate all trained models with IoU and F-score"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Always use 5 classes (ignoring background)
    num_classes = 5
    class_names_display = CLASS_NAMES_5CLS
    print(f"Class mode: 5 classes (ignoring background, plant, sea-floor)\n")
    
    # Load test dataset with merge_classes=True to ignore background
    test_dataset = SUIMDataset(test_file, transform=val_transforms, merge_classes=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=4)
    print(f"Test set: {len(test_dataset)} images\n")
    
    # Models to evaluate - all use 5 classes (ignoring background)
    class_mode = "5cls"
    keras_class_mode = "5cls"
    
    models_to_eval = [
        {
            'name': 'SUIM-Net (Keras-VGG)',
            'model_name': 'suimnet_keras_vgg',
            'checkpoint': f'checkpoints/suimnet_keras_vgg_{keras_class_mode}_noaug_best.weights.h5',
            'resolution': 256,
            'keras': True,
            'backbone': 'VGG',
            'im_res': (320, 256, 3)
        },
        {
            'name': 'SUIM-Net (Keras-RSB)',
            'model_name': 'suimnet_keras_rsb',
            'checkpoint': f'checkpoints/suimnet_keras_rsb_{keras_class_mode}_noaug_best.weights.h5',
            'resolution': 240,
            'keras': True,
            'backbone': 'RSB',
            'im_res': (320, 240, 3)
        },
        {
            'name': 'SUIM-Net (PyTorch)',
            'model_name': 'suimnet',
            'checkpoint': 'checkpoints/suimnet_6cls_aug_best.pth',
            'resolution': 256
        },
        {
            'name': 'UNet-ResAttn',
            'model_name': 'unet_resattn',
            'checkpoint': 'checkpoints/unet_resattn_6cls_aug_best.pth',
            'resolution': 256
        },
        {
            'name': 'UNet-ResAttn-V2',
            'model_name': 'unet_resattn_v2',
            'checkpoint': f'checkpoints/unet_resattn_v2_{class_mode}_aug_best.pth',
            'resolution': 256
        },
        {
            'name': 'UNet-ResAttn-V3',
            'model_name': 'unet_resattn_v3',
            'checkpoint': f'checkpoints/unet_resattn_v3_{class_mode}_aug_best.pth',
            'resolution': 384
        },
        {
            'name': 'UNet-ResAttn-V4',
            'model_name': 'unet_resattn_v4',
            'checkpoint': 'checkpoints/unet_resattn_v4_6cls_aug_best.pth',
            'resolution': 384
        },
        {
            'name': 'DeepLabV3-ResNet50',
            'model_name': 'deeplabv3',
            'checkpoint': 'checkpoints/deeplabv3_6cls_aug_best.pth',
            'resolution': 256
        },
        {
            'name': 'UWSegFormer',
            'model_name': 'uwsegformer',
            'checkpoint': 'checkpoints/uwsegformer_6cls_aug_best.pth',
            'resolution': 384
        },
        {
            'name': 'UWSegFormer-V2',
            'model_name': 'uwsegformer_v2',
            'checkpoint': f'checkpoints/uwsegformer_v2_{class_mode}_aug_best.pth',
            'resolution': 384
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
            # Handle Keras models separately
            if model_info.get('keras', False):
                # Load Keras model
                model = load_keras_model(
                    model_info['backbone'],
                    model_info['checkpoint'],
                    num_classes=5,  # Keras models use 5 classes
                    im_res=model_info['im_res']
                )
                
                # Evaluate Keras model
                metrics = evaluate_keras_model(
                    model, 
                    test_loader, 
                    num_classes=5,
                    im_res=(model_info['im_res'][0], model_info['im_res'][1])
                )
            else:
                # Load PyTorch model (use 5 classes for evaluation)
                model = load_model(
                    model_info['model_name'],
                    model_info['checkpoint'],
                    device,
                    num_classes=5
                )
                
                # Evaluate PyTorch model
                metrics = evaluate_model_full(model, test_loader, device, num_classes=5)
            
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
    
    for i, class_name in enumerate(class_names_display):
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
    
    for i, class_name in enumerate(class_names_display):
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
    parser = argparse.ArgumentParser(description='Evaluate all models with IoU and F-score')
    parser.add_argument('--test_file', type=str, default='data/test.txt',
                       help='Path to test split file')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for evaluation')
    parser.add_argument('--merge-classes', action='store_true', default=True,
                       help='Always True: evaluates with 5 classes (ignoring background, plant, sea-floor)')
    
    args = parser.parse_args()
    
    # Always evaluate with 5 classes (merge_classes=True)
    evaluate_all_models(args.test_file, args.batch_size, merge_classes=True)
