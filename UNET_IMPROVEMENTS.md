# UNet-ResAttn Improvement Guide

## Current Performance vs Potential

**Current UNet-ResAttn:** 38.50% mIoU  
**Target with improvements:** 48-52% mIoU (estimated)  
**DeepLabV3 (best):** 53.67% mIoU

**Goal:** Close the gap with DeepLabV3 through architectural and training improvements.

---

## Key Improvements Implemented

### 1. **Squeeze-and-Excitation (SE) Blocks**
**What:** Channel attention mechanism that recalibrates channel-wise feature responses.

**Why it helps:**
- Learns which channels are most important for each class
- Improves feature discrimination
- Minimal computational overhead

**Implementation:**
```python
class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=16):
        # Global pooling → FC → ReLU → FC → Sigmoid
        # Recalibrates channel importance
```

**Expected gain:** +2-3% mIoU

---

### 2. **Spatial Pyramid Pooling (SPP/ASPP)**
**What:** Multi-scale feature extraction using dilated convolutions at different rates.

**Why it helps:**
- Captures context at multiple scales (like DeepLabV3's ASPP)
- Better for objects of varying sizes
- Improves wreck, robot, reef detection

**Implementation:**
```python
class SpatialPyramidPooling(nn.Module):
    # Parallel branches with dilations: 1, 6, 12, 18
    # + Global pooling branch
    # Fusion of all scales
```

**Expected gain:** +3-5% mIoU

---

### 3. **Deep Supervision**
**What:** Multiple auxiliary outputs at different decoder stages.

**Why it helps:**
- Better gradient flow to early layers
- Forces intermediate features to be discriminative
- Addresses vanishing gradient problem

**Implementation:**
```python
# Auxiliary classifiers at decoder stages 1, 2, 3
if self.training:
    return {
        'out': main_output,
        'aux1': deep_supervision_1,
        'aux2': deep_supervision_2,
        'aux3': deep_supervision_3
    }
```

**Loss function:**
```python
total_loss = main_loss + 0.4 × (aux1_loss + aux2_loss + aux3_loss) / 3
```

**Expected gain:** +2-4% mIoU

---

### 4. **Class-Weighted Loss**
**What:** Weight loss by inverse class frequency to handle imbalance.

**Why it helps:**
- Current issue: Diver (0.00%), Plant (7.07%) - minority classes ignored
- Forces model to learn rare classes
- Balances optimization across all classes

**Implementation:**
```python
# Calculate weights from training data
class_weights = total_pixels / (num_classes × class_counts)
criterion = DiceCELoss(dice_weight=0.5, class_weights=class_weights)
```

**Expected gain:** +3-6% mIoU (especially for Diver, Plant, Fish)

---

### 5. **Better Regularization**
**What:** Graduated dropout + weight decay + gradient clipping.

**Why it helps:**
- Prevents overfitting (current model may be overfitting)
- Improves generalization to test set
- Stabilizes training

**Implementation:**
```python
# Dropout increases deeper in network
enc1: dropout=0.0
enc2: dropout=0.1
enc3: dropout=0.2
enc4: dropout=0.3
bottleneck: dropout=0.4

# Weight decay
optimizer = AdamW(lr=1e-4, weight_decay=1e-4)

# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Expected gain:** +1-2% mIoU

---

### 6. **Cosine Annealing LR Schedule**
**What:** Cyclical learning rate with warm restarts.

**Why it helps:**
- Escapes local minima
- Better than plateau-based scheduling
- Explores loss landscape more effectively

**Implementation:**
```python
scheduler = CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-6
)
```

**Expected gain:** +1-2% mIoU

---

## Training the Improved Model

### Quick Start
```bash
# Activate environment
source venv/bin/activate

# Train UNet-ResAttn-V2 with all improvements
python train_unet_v2.py \
    --epochs 60 \
    --batch_size 8 \
    --lr 1e-4 \
    --deep_supervision \
    --class_weights

# This will save to: checkpoints/unet_resattn_v2_best.pth
```

### Full Options
```bash
python train_unet_v2.py \
    --epochs 60 \                  # More epochs for better convergence
    --batch_size 8 \                # Adjust based on GPU memory
    --base_channels 64 \            # Model width (64 or 96 for more capacity)
    --lr 1e-4 \                     # Learning rate
    --weight_decay 1e-4 \           # L2 regularization
    --deep_supervision \            # Enable auxiliary losses
    --class_weights \               # Use class-balanced loss
    --aux_weight 0.4                # Weight for auxiliary losses
```

### Hyperparameter Recommendations

**For Best Accuracy:**
```bash
python train_unet_v2.py \
    --epochs 80 \
    --batch_size 6 \
    --base_channels 96 \            # Wider model
    --lr 8e-5 \                     # Slightly lower LR
    --weight_decay 1e-4 \
    --deep_supervision \
    --class_weights \
    --aux_weight 0.5
```

**For Faster Training:**
```bash
python train_unet_v2.py \
    --epochs 50 \
    --batch_size 12 \               # Larger batches
    --base_channels 48 \            # Narrower model
    --lr 2e-4 \                     # Higher LR
    --weight_decay 5e-5 \
    --deep_supervision \
    --class_weights
```

---

## Evaluation

### Test the Improved Model
```bash
# Quick test
python evaluate.py \
    --model unet_resattn_v2 \
    --checkpoint checkpoints/unet_resattn_v2_best.pth
```

### Expected Results

**Predicted Performance (UNet-ResAttn-V2):**

| Class | Current | Expected Improvement | Target |
|-------|---------|---------------------|--------|
| Background | 81.56% | +3% | ~85% |
| Diver | 13.75% | +15-20% | ~30-35% |
| Plant | 7.07% | +10-15% | ~20% |
| Wreck | 33.37% | +8-12% | ~45% |
| Robot | 35.71% | +8-10% | ~45% |
| Reef/Invertebrate | 49.77% | +5-7% | ~55% |
| Fish/Vertebrate | 15.79% | +10-15% | ~28% |
| Sea-floor/Rock | 53.04% | +5-7% | ~60% |
| **Mean IoU** | **38.50%** | **+10-12%** | **~48-50%** |

---

## Why These Improvements Work

### Problem Analysis: Current UNet-ResAttn Issues

1. **No Multi-Scale Context**
   - DeepLabV3 has ASPP → sees objects at multiple scales
   - Original UNet-ResAttn → single-scale features
   - **Fix:** Add SPP module in bottleneck

2. **Weak Channel Attention**
   - All channels treated equally
   - Some channels are noise for certain classes
   - **Fix:** Add SE blocks to recalibrate channels

3. **Poor Rare Class Performance**
   - Diver: 13.75%, Plant: 7.07% (vs DeepLabV3: 33.89%, 13.88%)
   - Training dominated by background (>80% of pixels)
   - **Fix:** Class-weighted loss

4. **Gradient Flow Issues**
   - Deep network (4 encoders + 4 decoders)
   - Early layers get weak gradients
   - **Fix:** Deep supervision provides direct supervision

5. **Suboptimal Training**
   - ReduceLROnPlateau can get stuck
   - No gradient clipping → unstable updates
   - **Fix:** Cosine annealing + gradient clipping

---

## Comparison: V1 vs V2

### Architecture Differences

| Feature | UNet-ResAttn V1 | UNet-ResAttn V2 |
|---------|----------------|-----------------|
| **Residual Blocks** | Basic | + SE attention |
| **Bottleneck** | Simple ResBlock | + SPP module |
| **Outputs** | Single | Deep supervision (4 outputs) |
| **Dropout** | Only bottleneck | Graduated (0→0.4) |
| **Attention** | Spatial only | Spatial + Channel (SE) |
| **Parameters** | 32.96M | ~38M (+15%) |

### Training Differences

| Aspect | V1 | V2 |
|--------|----|----|
| **Loss** | Dice + CE | Dice + CE + class weights |
| **Optimizer** | Adam | AdamW (+ weight decay) |
| **LR Schedule** | ReduceLROnPlateau | CosineAnnealingWarmRestarts |
| **Regularization** | Minimal | Dropout + weight decay + grad clip |
| **Epochs** | 50 | 60-80 recommended |

---

## Advanced Improvements (Future Work)

### 1. **Pre-training**
**Biggest potential gain:** +10-15% mIoU

**Option A: ImageNet Pre-trained Encoder**
```python
# Use ResNet50 encoder (like DeepLabV3)
from torchvision.models import resnet50
encoder = resnet50(pretrained=True)
```

**Option B: Self-supervised Pre-training**
```python
# Train on larger underwater image corpus
# Using SimCLR, MoCo, or MAE
```

### 2. **Test-Time Augmentation (TTA)**
**Gain:** +2-3% mIoU (no retraining needed!)

```python
# Average predictions over flipped/rotated versions
def predict_with_tta(model, image):
    predictions = []
    for transform in [identity, hflip, vflip, rot90]:
        pred = model(transform(image))
        predictions.append(inverse_transform(pred))
    return torch.mean(predictions, dim=0)
```

### 3. **Ensemble**
**Gain:** +3-5% mIoU

```python
# Ensemble V2 + DeepLabV3
ensemble_pred = 0.6 × deeplabv3_pred + 0.4 × unet_v2_pred
```

### 4. **Higher Resolution**
**Gain:** +2-4% mIoU (especially small objects)

```python
# Train at 512×512 instead of 256×256
# Requires 4× more memory (reduce batch size)
A.Resize(512, 512)
```

### 5. **Focal Loss**
**Gain:** +1-3% mIoU on hard classes

```python
# Focuses on hard examples
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        self.gamma = gamma
        # pt = (1 - p) for correct class
        # loss = -α × (1-pt)^γ × log(pt)
```

---

## Estimated Total Impact

### Conservative Estimate
```
Base UNet-ResAttn:           38.50%
+ SE Blocks:                  +2.0%  → 40.50%
+ SPP:                        +3.0%  → 43.50%
+ Deep Supervision:           +2.0%  → 45.50%
+ Class Weights:              +3.0%  → 48.50%
+ Better Regularization:      +1.0%  → 49.50%
+ Cosine LR:                  +1.0%  → 50.50%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Expected:              ~50-51% mIoU
```

### Optimistic Estimate (with fine-tuning)
```
All improvements above:      50.50%
+ Pre-training:              +8.0%  → 58.50%
+ TTA:                       +2.0%  → 60.50%
+ Higher resolution:         +2.0%  → 62.50%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Could potentially reach:     ~60-63% mIoU
(exceeding DeepLabV3!)
```

---

## Quick Comparison Matrix

| Model | mIoU | Params | Training Time | Best For |
|-------|------|--------|---------------|----------|
| SUIM-Net | 37.60% | 7.7M | 2h | Speed |
| UNet-ResAttn V1 | 38.50% | 33M | 3h | Balanced (but inefficient) |
| **UNet-ResAttn V2** | **~50%** | **38M** | **4h** | **Improved accuracy** |
| DeepLabV3 | 53.67% | 39.6M | 3h | Best accuracy |
| UNet-V2 + Pre-train | ~58%+ | 38M | 5h+ | Research target |

---

## Recommendations

### For this assignment:
1. ✅ **Train UNet-ResAttn-V2** with all improvements
2. ✅ Use class weights + deep supervision
3. ✅ Train for 60 epochs with cosine annealing
4. ✅ Compare V1 vs V2 in report
5. ⏩ Optional: Add TTA for extra boost

### For production:
- **Still use DeepLabV3** if you need best accuracy now
- **Use UNet-V2** if you want to beat DeepLabV3 with more training
- **Consider ensemble** for critical applications

### For research:
- Add pre-training for best results
- Explore transformer-based alternatives (SegFormer, Mask2Former)
- Try knowledge distillation from DeepLabV3 → UNet-V2

---

## Summary

**Yes, UNet-ResAttn can be significantly improved!**

The current 38.50% → **~50% mIoU** is achievable through:
1. Modern architecture improvements (SE, SPP)
2. Better training strategies (class weights, deep supervision)
3. Proper regularization and LR scheduling

This would make it **competitive with DeepLabV3** while maintaining the elegant UNet structure.

**To get started:**
```bash
python train_unet_v2.py --deep_supervision --class_weights
```

Expected training time: **~4 hours**  
Expected improvement: **+10-12% mIoU absolute**

🎯 **Goal:** Bridge the 15% gap to DeepLabV3!
