# UNet-ResAttn-V3: Strategic Improvements

## Overview

UNet-ResAttn-V3 is designed with **high-impact, proven improvements** based on the lessons learned from V1 and V2:

- **V1** (38.50% mIoU): Good baseline but lacks pre-training
- **V2** (35.64% mIoU): Too complex, unstable training, overfitting
- **V3** (Target: 48-52% mIoU): Strategic improvements only

## Key Improvements

### 1. ⭐⭐⭐⭐⭐ Pre-trained ResNet-50 Encoder (BIGGEST IMPACT)
**Why it works:**
- ResNet-50 trained on ImageNet provides strong feature representations
- This is why DeepLabV3 achieves 53.67% mIoU
- Expected gain: **+10-15% mIoU**

**Implementation:**
- Use `torchvision.models.resnet50(pretrained=True)`
- Extract features from layer1, layer2, layer3, layer4
- Lower learning rate for encoder (0.1x) vs decoder (1.0x)

### 2. ⭐⭐⭐⭐ Focal Loss Instead of Class-Weighted CE
**Why it works:**
- Focuses on hard-to-classify examples
- Better than class weights for extreme imbalance
- Diver class (weight=5.89) needs special attention
- Expected gain: **+3-5% mIoU** especially on rare classes

**Implementation:**
- DiceFocalLoss with gamma=2.0
- Alpha parameter set to class weights
- Focal weight: (1 - pt)^gamma emphasizes hard examples

### 3. ⭐⭐⭐ Higher Resolution (384×384)
**Why it works:**
- Small objects (divers) lost at 256×256
- More spatial detail for precise segmentation
- Expected gain: **+2-4% mIoU**

**Trade-off:**
- More GPU memory (batch_size 6 instead of 8)
- ~1.5x training time

### 4. ⭐⭐⭐ Lightweight SE Blocks
**Why it works:**
- Channel attention helps without massive parameter overhead
- Proven effective in SENet, EfficientNet
- Expected gain: **+1-2% mIoU**

**What we learned from V2:**
- Don't overdo it (V2 had too many features)
- SE blocks in decoder only (encoder is pre-trained)
- No deep supervision (causes training instability)
- No SPP (complexity without clear benefit)

## Architecture Details

```
Input: 384×384×3

Encoder (ResNet-50, pre-trained):
  enc1: 192×192×64   (conv1 + bn + relu)
  enc2: 96×96×256    (layer1, after maxpool)
  enc3: 48×48×512    (layer2)
  enc4: 24×24×1024   (layer3)
  enc5: 12×12×2048   (layer4)

Bridge:
  12×12×1024 (2 conv layers)

Decoder (with Attention + SE):
  dec4: 24×24×512    (+ enc4 via attention gate + SE)
  dec3: 48×48×256    (+ enc3 via attention gate + SE)
  dec2: 96×96×128    (+ enc2 via attention gate + SE)
  dec1: 192×192×64   (+ enc1 via attention gate + SE)

Output: 384×384×8
```

**Total Parameters: 74.5M**
- Encoder (ResNet-50): ~25M (frozen LR × 0.1)
- Decoder: ~49M (trainable LR × 1.0)

## Training Configuration

### Loss Function
- **DiceFocalLoss** (50% Dice + 50% Focal)
- Focal Loss parameters:
  - `gamma=2.0` (focus on hard examples)
  - `alpha=class_weights` (handle imbalance)

### Optimizer
- **AdamW** with differential learning rates:
  - Encoder: `1e-5` (0.1× base LR, pre-trained)
  - Decoder: `1e-4` (1.0× base LR, random init)
- Weight decay: `1e-4`
- Gradient clipping: `1.0`

### Learning Rate Schedule
- **ReduceLROnPlateau**
  - Monitor: validation mIoU
  - Factor: 0.5
  - Patience: 5 epochs
  - Min LR: 1e-6

### Data Augmentation (384×384)
- Resize to 384×384
- HorizontalFlip (p=0.5)
- VerticalFlip (p=0.5)
- ShiftScaleRotate (p=0.5)
- ColorJitter OR GaussianBlur (p=0.5)

### Training Details
- Epochs: 50
- Batch size: 6 (GPU memory limit at 384×384)
- Image size: 384×384 (vs 256×256 in V1/V2)
- Validation: Every epoch
- Early stopping: Save best based on validation mIoU

## Expected Performance

### Conservative Estimate
| Metric | V1 | V2 | V3 Target |
|--------|----|----|-----------|
| Test mIoU | 38.50% | 35.64% | **48-50%** |
| Background | 82.11% | 80.51% | **~85%** |
| Diver | 13.75% | 13.08% | **~28%** |
| Fish | 30.40% | 14.45% | **~38%** |
| Reefs | 48.39% | 31.75% | **~55%** |
| Plants | 7.07% | 32.01% | **~20%** |
| Wrecks | 50.22% | 43.61% | **~58%** |
| Robots | 28.75% | 15.45% | **~40%** |
| Rocks | 54.55% | 47.29% | **~60%** |

### Optimistic Estimate (if everything works well)
- **Test mIoU: 50-52%**
- Close to DeepLabV3 (53.67%)

## Why V3 Will Outperform V2

| Aspect | V2 (Failed) | V3 (Improved) |
|--------|-------------|---------------|
| **Encoder** | Random init | Pre-trained ResNet-50 |
| **Parameters** | 68.85M | 74.5M (but smarter) |
| **Complexity** | Too high (SPP + DS) | Just right (SE only) |
| **Loss** | Class-weighted CE | Focal Loss |
| **Training** | Cosine annealing | ReduceLROnPlateau |
| **Resolution** | 256×256 | 384×384 |
| **Stability** | Unstable (deep supervision) | Stable |
| **Overfitting** | High (39% val → 36% test) | Lower (differential LR) |

## Lessons Applied from V2 Failure

1. **Simpler is better**: Removed deep supervision and SPP
2. **Pre-training is key**: Added ResNet-50 encoder
3. **Stable training**: ReduceLROnPlateau instead of cosine annealing
4. **Right loss function**: Focal Loss > Class-weighted CE
5. **Appropriate complexity**: SE blocks only, not everything at once

## Training Command

```bash
python train_unet_v3.py --epochs 50 --batch_size 6 --lr 1e-4 --focal_gamma 2.0 --use_focal
```

**Expected training time:** ~6 hours (50 epochs × 7 min/epoch)

## Next Steps After Training

1. **Evaluate on test set**
2. **Compare with all models:**
   - SUIM-Net: 37.60%
   - UNet-ResAttn-V1: 38.50%
   - UNet-ResAttn-V2: 35.64%
   - **UNet-ResAttn-V3: ???**
   - DeepLabV3: 53.67%

3. **If V3 reaches 48-50%:**
   - Add Test-Time Augmentation (+2-3%)
   - Try ensemble with DeepLabV3 (+2-3%)
   - Could reach **52-55% mIoU**!

4. **If V3 underperforms (<45%):**
   - Fine-tune hyperparameters
   - Try EfficientNet encoder instead of ResNet-50
   - Increase training epochs to 80-100

## Files Created

- `models/unet_resattn_v3.py` - Model architecture
- `training/loss.py` - Added FocalLoss and DiceFocalLoss
- `train_unet_v3.py` - Training script
- `UNET_V3_STRATEGY.md` - This document

---

**Status:** Ready to train  
**Target:** 48-52% mIoU  
**Benchmark:** DeepLabV3 at 53.67% mIoU
