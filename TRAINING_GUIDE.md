# Training Guide - Underwater Semantic Segmentation

## Prerequisites

Before training, ensure you have:
1. ✅ Raw SUIM dataset in `raw_suim/` folder
2. ✅ Organized dataset structure (run setup scripts - see README.md)
3. ✅ Virtual environment activated with all dependencies
4. ✅ CUDA-enabled GPU (recommended)

---

## Quick Start - Train All Models

### Option 1: Run in Background with nohup (Recommended)
```bash
nohup bash train_all_models.sh > train_output.log 2>&1 &
```

This will:
- Run the script in the background
- Keep running even if you close the terminal
- Save output to `train_output.log`
- Save detailed logs to `logs/train_all_TIMESTAMP.log`

### Option 2: Run in a tmux/screen session
```bash
# Start a tmux session
tmux new -s training

# Run the script
bash train_all_models.sh

# Detach: Press Ctrl+B, then D
# Reattach later: tmux attach -t training
```

### Option 3: Simple foreground execution
```bash
bash train_all_models.sh
```

---

## What Gets Trained

The script trains **5 models** sequentially:

### 1. SUIM-Net (Lightweight Baseline)
```bash
python main_train.py --model suimnet --epochs 50 --batch_size 8 --augment --lr 1e-4
```
- **Parameters:** 7.76M
- **Training time:** ~2 hours
- **Expected mIoU:** ~33% (test set)
- **Checkpoint:** `checkpoints/suimnet_aug_best.pth`
- **Use case:** Real-time/edge deployment

### 2. UNet-ResAttn V1 (Custom Baseline)
```bash
python main_train.py --model unet_resattn --epochs 50 --batch_size 8 --augment --lr 1e-4
```
- **Parameters:** 32.96M
- **Training time:** ~3 hours
- **Expected mIoU:** ~36% (test set)
- **Checkpoint:** `checkpoints/unet_resattn_aug_best.pth`
- **Use case:** Baseline custom architecture

### 3. UNet-ResAttn V2 (Failed Experiment)
```bash
python train_unet_v2.py --epochs 60 --batch_size 8
```
- **Parameters:** 68.85M
- **Training time:** ~4 hours
- **Expected mIoU:** ~35% (test set) - **underperforms V1!**
- **Checkpoint:** `checkpoints/unet_resattn_v2_best.pth`
- **Note:** Over-engineered without pre-training, kept for research reference

### 4. UNet-ResAttn V3 (Best Model) 🏆
```bash
python train_unet_v3.py --epochs 50 --batch_size 6
```
- **Parameters:** 74.49M
- **Training time:** ~3.5 hours
- **Expected mIoU:** ~52% (test set) - **BEST PERFORMANCE!**
- **Expected F-score:** ~62%
- **Checkpoint:** `checkpoints/unet_resattn_v3_best.pth`
- **Features:** Pre-trained ResNet-50, Focal Loss, 384×384 resolution
- **Use case:** Maximum accuracy production deployment

### 5. DeepLabV3-ResNet50 (State-of-the-Art Baseline)
```bash
python main_train.py --model deeplabv3 --epochs 30 --batch_size 4 --augment --lr 1e-4
```
- **Parameters:** 39.64M
- **Training time:** ~3 hours
- **Expected mIoU:** ~51% (test set)
- **Expected F-score:** ~60%
- **Checkpoint:** `checkpoints/deeplabv3_aug_best.pth`
- **Use case:** Balanced performance/efficiency

**Total estimated time: 15-18 hours** for all 5 models (depending on GPU)

---

## Training Individual Models

### Train Only Best Model (V3)
```bash
source venv/bin/activate
python train_unet_v3.py --epochs 50 --batch_size 6
```

### Train Only DeepLabV3
```bash
source venv/bin/activate
python main_train.py --model deeplabv3 --epochs 30 --batch_size 4 --augment
```

### Skip Failed Experiments
If you want to skip V2 (the failed experiment), edit `train_all_models.sh` and comment out:
```bash
# echo "Training UNet-ResAttn-V2..."
# python train_unet_v2.py --epochs 60 --batch_size 8 || echo "V2 training failed"
```

---

## Monitor Progress

### Check if training is running:
```bash
ps aux | grep -E "train_all_models|train_unet|main_train"
```

### Monitor GPU usage:
```bash
watch -n 1 nvidia-smi
```

### Tail the log file:
```bash
tail -f logs/train_all_*.log
```

Or for nohup:
```bash
tail -f train_output.log
```

### Check latest checkpoint:
```bash
ls -lht checkpoints/*.pth | head -5
```

### Monitor specific model training:
```bash
# Watch for validation mIoU improvements
tail -f logs/train_all_*.log | grep "Val mIoU"

# Watch for epoch completion
tail -f logs/train_all_*.log | grep "Epoch"
```

---

## Stop Training

If you need to stop:

```bash
# Find the process ID
ps aux | grep -E "train_all_models|train_unet|main_train"

# Kill specific process (replace PID with actual process ID)
kill PID

# Or kill all Python training processes (WARNING: kills all)
pkill -f "python.*train"

# For tmux session
tmux kill-session -t training
```

---

## After Training Completes

### Check all results with F-score:
```bash
source venv/bin/activate
python evaluate_with_fscore.py
```

This evaluates all 5 models and generates:
- Per-class IoU and F-score
- Overall performance comparison
- Results saved to `evaluation_results_with_fscore.txt`

### Evaluate individual models:
```bash
# SUIM-Net
python evaluate.py --model suimnet --checkpoint checkpoints/suimnet_aug_best.pth

# UNet-ResAttn V1
python evaluate.py --model unet_resattn --checkpoint checkpoints/unet_resattn_aug_best.pth

# DeepLabV3
python evaluate.py --model deeplabv3 --checkpoint checkpoints/deeplabv3_aug_best.pth
```

### View comprehensive results:
```bash
# Check the final summary
cat evaluation_results_with_fscore.txt

# View training report
cat TRAINING_REPORT.md

# View final results
cat FINAL_RESULTS.md
```

---

## Expected Results Summary

| Model | Test mIoU | Test F-score | Parameters | Status |
|-------|-----------|--------------|------------|--------|
| **UNet-ResAttn-V3** | **51.91%** | **61.52%** | 74.49M | 🏆 **BEST** |
| DeepLabV3-ResNet50 | 50.65% | 59.75% | 39.64M | ⭐ Strong baseline |
| UNet-ResAttn V1 | 36.26% | 45.75% | 32.96M | Baseline custom |
| UNet-ResAttn V2 | 34.77% | 44.84% | 68.85M | ❌ Failed experiment |
| SUIM-Net | 33.12% | 41.55% | 7.76M | Lightweight |

**Winner:** UNet-ResAttn-V3 (pre-trained ResNet-50 + Focal Loss + 384×384)

---

## Troubleshooting

### Out of Memory Error:
```bash
# Reduce batch size for specific models
# Edit train_unet_v3.py and change:
python train_unet_v3.py --epochs 50 --batch_size 4  # instead of 6

# Or for DeepLabV3:
python main_train.py --model deeplabv3 --epochs 30 --batch_size 2  # instead of 4
```

### Training too slow on CPU:
- The script automatically uses GPU if available
- Training on CPU is **NOT recommended** (10-20x slower)
- Expected times are for GPU (NVIDIA RTX 3060 or similar)

### CUDA Out of Memory:
```bash
# Clear GPU memory
nvidia-smi

# Kill any zombie processes
pkill -f python

# Restart training with smaller batch size
```

### Checkpoint loading errors:
```bash
# Check if checkpoint exists
ls -lh checkpoints/

# Verify checkpoint integrity
python -c "import torch; print(torch.load('checkpoints/unet_resattn_v3_best.pth', weights_only=False).keys())"
```

### Need to resume interrupted training:
- Currently, scripts start from scratch
- Checkpoints are saved each epoch (best validation)
- To resume: modify training scripts to load checkpoint and continue

### ValueError: RGB palette errors:
- Ensure you're using corrected palette (255, not 128)
- Check `datasets/suim_dataset.py` has correct `RGB_TO_CLASS` mapping
- Verify with: `python -c "from datasets.suim_dataset import RGB_TO_CLASS; print(RGB_TO_CLASS)"`

---

## Customization

### Change Training Parameters

Edit individual training scripts or `train_all_models.sh`:

```bash
# Longer training
python train_unet_v3.py --epochs 75 --batch_size 6

# Different learning rate
python main_train.py --model deeplabv3 --lr 5e-5

# Without augmentation (not recommended)
python main_train.py --model suimnet --epochs 50 --batch_size 8

# Custom class weights for V3
# Edit train_unet_v3.py and modify get_class_weights() function
```

### Train Specific Model Versions

```bash
# Only train best models (V3 + DeepLabV3)
python train_unet_v3.py --epochs 50 --batch_size 6
python main_train.py --model deeplabv3 --epochs 30 --batch_size 4 --augment

# Quick test run (few epochs)
python train_unet_v3.py --epochs 5 --batch_size 6

# Experiment with Focal Loss gamma
python train_unet_v3.py --epochs 50 --batch_size 6 --focal_gamma 3.0
```

### Modify Data Augmentation

Edit augmentation pipeline in training scripts:
- `main_train.py` - For SUIM-Net, V1, DeepLabV3
- `train_unet_v2.py` - For V2
- `train_unet_v3.py` - For V3

Example modifications:
```python
# Increase augmentation intensity
train_transform = A.Compose([
    A.Resize(384, 384),
    A.HorizontalFlip(p=0.7),  # increased from 0.5
    A.VerticalFlip(p=0.3),    # increased from 0.2
    A.RandomRotate90(p=0.7),  # increased from 0.5
    # ... add more augmentations
])
```

---

## Advanced: Distributed Training

For multi-GPU training (future enhancement):

```bash
# Not yet implemented, but could be added:
python -m torch.distributed.launch --nproc_per_node=2 train_unet_v3.py
```

---

## Performance Benchmarks

On NVIDIA RTX 3060 (12GB):

| Model | Batch Size | Time/Epoch | Total Time (50 epochs) | GPU Memory |
|-------|-----------|------------|------------------------|------------|
| SUIM-Net | 8 | 2-3 min | ~2 hours | ~4 GB |
| UNet-V1 | 8 | 3-4 min | ~3 hours | ~6 GB |
| UNet-V2 | 8 | 4-5 min | ~4 hours | ~8 GB |
| UNet-V3 | 6 | 4-5 min | ~3.5 hours | ~10 GB |
| DeepLabV3 | 4 | 4-5 min | ~3 hours (30 epochs) | ~9 GB |

**Note:** V3 uses 384×384 resolution (others use 256×256), requiring more memory.

---

## Files Generated During Training

```
CV_Assign2/
├── checkpoints/
│   ├── suimnet_aug_best.pth           (~89 MB)
│   ├── unet_resattn_aug_best.pth      (~378 MB)
│   ├── unet_resattn_v2_best.pth       (~790 MB)
│   ├── unet_resattn_v3_best.pth       (~859 MB) 🏆
│   └── deeplabv3_aug_best.pth         (~464 MB)
├── logs/
│   └── train_all_TIMESTAMP.log         (training logs)
├── train_output.log                    (if using nohup)
└── evaluation_results_with_fscore.txt  (after evaluation)
```

**Total checkpoint size:** ~2.5 GB

---

## Next Steps After Training

1. **Evaluate all models:**
   ```bash
   python evaluate_with_fscore.py
   ```

2. **Read comprehensive reports:**
   - `TRAINING_REPORT.md` - Detailed analysis of all models
   - `FINAL_RESULTS.md` - Results summary
   - `STATE_OF_THE_ART.md` - Performance benchmarks

3. **Use best model for inference:**
   - Load `checkpoints/unet_resattn_v3_best.pth`
   - 384×384 input resolution
   - See inference examples in repository

4. **Optional improvements:**
   - Test-time augmentation (+2-3% mIoU)
   - Ensemble V3 + DeepLabV3 (+2-3% mIoU)
   - Longer training (75-100 epochs)

---

**Training Guide Last Updated:** November 25, 2025
