# CV Assignment 2 – Proposal 6: Underwater Semantic Segmentation (SUIM)

## Overview
Semantic segmentation of underwater images using the SUIM dataset (1,635 images, 8 classes).

**Models:**
- **UNet-ResAttn** (Custom): UNet + Residual blocks + Attention gates (33M params)
- **SUIM-Net** (Baseline): Paper-inspired lightweight encoder-decoder (7.8M params)
- **DeepLabV3-ResNet50** (Baseline): Pre-trained torchvision model (40M params)

**Classes:** Background, Diver, Plant, Wreck, Robot, Reef/Invertebrate, Fish/Vertebrate, Sea-floor

---

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Dataset Organization
Dataset already organized:
- `data/images/` - 1,635 images
- `data/masks/` - 1,635 masks
- `data/train.txt` - 1,220 training image IDs
- `data/val.txt` - 305 validation image IDs
- `data/test.txt` - 110 test image IDs

---

## Quick Test (Verify Models Work)
```bash
python test_models.py
```
Tests all 3 models with dummy data (no GPU needed, ~10 seconds).

---

## Training

### Train UNet-ResAttn with augmentation
```bash
python main_train.py --model unet_resattn --epochs 50 --batch_size 8 --augment
```

### Train SUIM-Net without augmentation
```bash
python main_train.py --model suimnet --epochs 50 --batch_size 8
```

### Train DeepLabV3
```bash
python main_train.py --model deeplabv3 --epochs 30 --batch_size 4 --augment
```

**Key Arguments:**
- `--model`: `unet_resattn`, `suimnet`, or `deeplabv3`
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 8, reduce if OOM)
- `--augment`: Enable data augmentation
- `--lr`: Learning rate (default: 1e-4)

**Checkpoints saved to:** `checkpoints/{model}_aug_best.pth` or `checkpoints/{model}_noaug_best.pth`

---

## Evaluation

Run evaluation on test set:
```bash
python evaluate.py --model unet_resattn --checkpoint checkpoints/unet_resattn_aug_best.pth
```

---

## Jupyter Notebook (Alternative)
```bash
jupyter notebook notebook/UnderwaterSegmentation.ipynb
```
Interactive notebook with full pipeline: data loading, training, evaluation, visualization.

---

## Project Structure
```
├── data/
│   ├── images/          # All images
│   ├── masks/           # All masks
│   ├── train.txt        # Training split
│   ├── val.txt          # Validation split
│   └── test.txt         # Test split
├── models/
│   ├── unet_resattn.py  # Custom UNet-ResAttn
│   ├── suimnet.py       # SUIM-Net baseline
│   ├── deeplab_resnet.py # DeepLabV3 baseline
│   └── blocks/          # Residual & Attention modules
├── datasets/
│   ├── suim_dataset.py  # Dataset loader
│   └── augmentations.py # Data augmentation
├── training/
│   ├── train.py         # Training loop
│   ├── eval.py          # Evaluation metrics
│   ├── loss.py          # Dice + CE loss
│   └── utils.py         # Checkpoints & utilities
├── main_train.py        # Main training script
├── test_models.py       # Quick model test
└── requirements.txt     # Dependencies
```

---

## Hardware Requirements
- **GPU**: Highly recommended (6-8GB VRAM)
- **CPU only**: Very slow (10-20+ hours per model)
- **RAM**: 8-16GB minimum

**Recommended:** Use Google Colab or similar GPU platform for training.

---

## Results
After training, compare:
- mIoU (mean Intersection over Union)
- Per-class IoU
- Parameter count
- Training time
- With/without augmentation

---

## References
- SUIM Dataset: https://irvlab.cs.umn.edu/resources/suim-dataset
- Paper: https://arxiv.org/pdf/2004.01241.pdf
