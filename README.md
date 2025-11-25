# CV Assignment 2 – Underwater Semantic Segmentation (SUIM)

## Overview

Semantic segmentation of underwater images using the SUIM dataset (1,635 images, 8 classes).

**🏆 Best Model: UNet-ResAttn-V3** (51.91% mIoU, 61.52% F-score)

### Models Trained (5 total):

1. **UNet-ResAttn-V3** (Best): Pre-trained ResNet-50 + Focal Loss + 384×384 (74.5M params) - **51.91% mIoU**
2. **DeepLabV3-ResNet50**: Pre-trained state-of-the-art baseline (39.6M params) - **50.65% mIoU**
3. **UNet-ResAttn-V1**: Custom UNet + Residual + Attention (33.0M params) - **36.26% mIoU**
4. **UNet-ResAttn-V2**: Over-engineered experiment (68.9M params) - **34.77% mIoU** (failed)
5. **SUIM-Net**: Lightweight encoder-decoder (7.8M params) - **33.12% mIoU**

**Classes:** Background, Diver, Plant, Wreck, Robot, Reef/Invertebrate, Fish/Vertebrate, Sea-floor

---

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/ghostcat-pro/CV_Assign2.git
cd CV_Assign2
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download SUIM Dataset
Download the SUIM dataset from: https://irvlab.cs.umn.edu/resources/suim-dataset

You should have:
- `train_val/` folder with images and masks
- `TEST/` folder with test images and masks

### 5. Setup Dataset Structure

**IMPORTANT:** Manually place the raw SUIM data in the `raw_suim/` folder, then run the setup scripts.

```bash
# Create raw_suim directory
mkdir -p raw_suim

# Place your downloaded SUIM data here:
# raw_suim/
# ├── train_val/
# │   ├── images/     # Training + validation images
# │   └── masks/      # Training + validation masks
# └── TEST/
#     ├── images/     # Test images
#     └── masks/      # Test masks
```

After manually copying the data to `raw_suim/`, run the organization scripts:

```bash
# Step 1: Organize raw data into single folders
python organize_suim_dataset.py
```

This creates:
```
data/
├── all_images/     # All 1,635 images (combined train_val + TEST)
└── all_masks/      # All 1,635 masks (combined train_val + TEST)
```

```bash
# Step 2: Create train/val/test splits and copy files
python create_splits.py
```

This creates the final structure:
```
data/
├── images/
│   ├── train/      # 1,220 training images
│   ├── val/        # 305 validation images
│   └── test/       # 110 test images
├── masks/
│   ├── train/      # 1,220 training masks
│   ├── val/        # 305 validation masks
│   └── test/       # 110 test masks
├── train.txt       # List of training image IDs
├── val.txt         # List of validation image IDs
└── test.txt        # List of test image IDs
```

**Note:** The `data/`, `checkpoints/`, and `logs/` directories are in `.gitignore` because they contain large files. You must generate them locally using the scripts above.

---

## Verify Setup

### Test that models load correctly:
```bash
python test_models.py
```
This tests all 5 models with dummy data (no GPU needed, ~15 seconds).

### Verify dataset organization:
```bash
# Check that files exist
ls data/images/train/ | wc -l  # Should show 1220
ls data/images/val/ | wc -l    # Should show 305
ls data/images/test/ | wc -l   # Should show 110

# Check that split files exist
cat data/train.txt | wc -l     # Should show 1220
cat data/val.txt | wc -l       # Should show 305
cat data/test.txt | wc -l      # Should show 110
```

---

## Training

### Option 1: Train All Models (Recommended for Complete Results)

```bash
# Activate virtual environment
source venv/bin/activate

# Run training script (takes 15-18 hours)
bash train_all_models.sh
```

This trains all 5 models sequentially. See `TRAINING_GUIDE.md` for detailed instructions on:
- Running in background (nohup)
- Using tmux/screen sessions
- Monitoring progress
- Troubleshooting

### Option 2: Train Only Best Model (V3)

```bash
source venv/bin/activate
python train_unet_v3.py --epochs 50 --batch_size 6
```

**Training time:** ~3.5 hours on NVIDIA RTX 3060  
**Expected result:** ~52% mIoU, ~62% F-score

### Option 3: Train Individual Models

```bash
# SUIM-Net (lightweight baseline)
python main_train.py --model suimnet --epochs 50 --batch_size 8 --augment

# UNet-ResAttn V1 (custom baseline)
python main_train.py --model unet_resattn --epochs 50 --batch_size 8 --augment

# UNet-ResAttn V2 (failed experiment - optional)
python train_unet_v2.py --epochs 60 --batch_size 8

# UNet-ResAttn V3 (best model)
python train_unet_v3.py --epochs 50 --batch_size 6

# DeepLabV3 (state-of-the-art baseline)
python main_train.py --model deeplabv3 --epochs 30 --batch_size 4 --augment
```

**Key Training Arguments:**
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (reduce if out of memory)
- `--augment`: Enable data augmentation (recommended)
- `--lr`: Learning rate (default: 1e-4)
- `--focal_gamma`: Focal loss gamma for V3 (default: 2.0)

**Checkpoints saved to:** `checkpoints/{model}_best.pth`

---

## Evaluation

### Evaluate All Models with IoU and F-score:
```bash
source venv/bin/activate
python evaluate_with_fscore.py
```

This generates comprehensive results including:
- Mean IoU and F-score for each model
- Per-class IoU and F-score
- Performance comparison table
- Results saved to `evaluation_results_with_fscore.txt`

### Evaluate Individual Model:
```bash
# Example: Evaluate UNet-ResAttn-V3
python evaluate.py --model unet_resattn --checkpoint checkpoints/unet_resattn_v3_best.pth

# Example: Evaluate DeepLabV3
python evaluate.py --model deeplabv3 --checkpoint checkpoints/deeplabv3_aug_best.pth
```

---

## Results Summary

### Best Model Performance (Test Set)

| Model | mIoU | F-score | Parameters | Key Features |
|-------|------|---------|------------|--------------|
| **UNet-ResAttn-V3** 🏆 | **51.91%** | **61.52%** | 74.49M | Pre-trained ResNet-50 + Focal Loss + 384×384 |
| DeepLabV3-ResNet50 | 50.65% | 59.75% | 39.64M | Pre-trained + ASPP |
| UNet-ResAttn-V1 | 36.26% | 45.75% | 32.96M | Residual + Attention |
| UNet-ResAttn-V2 | 34.77% | 44.84% | 68.85M | Over-engineered (failed) |
| SUIM-Net | 33.12% | 41.55% | 7.76M | Lightweight baseline |

### Key Achievements

- ✅ **UNet-ResAttn-V3 beats DeepLabV3** by +1.26% mIoU
- ✅ **Exceeds original SUIM paper** baseline (~48% mIoU) by +3.91%
- ✅ **Wins on 5/8 classes** including critical Diver class (+6.58% over DeepLabV3)
- ✅ **A-grade academic performance** (51.91% in "Very Good" range for underwater)

### Per-Class Performance (V3 vs DeepLabV3)

| Class | V3 IoU | DeepLabV3 IoU | Winner |
|-------|--------|---------------|--------|
| Diver | **40.47%** | 33.89% | V3 (+6.58%) 🎯 |
| Wreck | **61.18%** | 55.13% | V3 (+6.05%) |
| Robot | **56.57%** | 53.06% | V3 (+3.51%) |
| Background | 85.93% | **86.93%** | DeepLabV3 |
| Reef | 58.67% | **57.61%** | V3 (+1.06%) |
| Fish | 37.80% | **41.04%** | DeepLabV3 |
| Plant | 15.30% | **13.88%** | V3 (+1.42%) |
| Sea-floor | 59.33% | **63.69%** | DeepLabV3 |

**For detailed analysis, see:**
- `TRAINING_REPORT.md` - Comprehensive training documentation
- `FINAL_RESULTS.md` - Results summary and analysis
- `STATE_OF_THE_ART.md` - Benchmark comparison
- `V3_UPDATE_SUMMARY.md` - V3 improvement breakdown

---

## Documentation

### Main Reports
- **`README.md`** (this file) - Quick start guide
- **`TRAINING_GUIDE.md`** - Detailed training instructions
- **`TRAINING_REPORT.md`** - Comprehensive analysis of all 5 models
- **`FINAL_RESULTS.md`** - Results summary and recommendations
- **`STATE_OF_THE_ART.md`** - mIoU benchmarks and industry standards
- **`UNET_V3_STRATEGY.md`** - V3 improvement strategy
- **`V3_UPDATE_SUMMARY.md`** - V3 update and impact analysis

### Technical Documentation
- Model implementations in `models/`
- Dataset loader in `datasets/`
- Training utilities in `training/`
- Evaluation scripts: `evaluate.py`, `evaluate_with_fscore.py`

---

## Project Structure

```
CV_Assign2/
├── data/                           # Dataset (generated by scripts)
│   ├── images/
│   │   ├── train/                  # 1,220 training images
│   │   ├── val/                    # 305 validation images
│   │   └── test/                   # 110 test images
│   ├── masks/
│   │   ├── train/                  # 1,220 training masks
│   │   ├── val/                    # 305 validation masks
│   │   └── test/                   # 110 test masks
│   ├── train.txt                   # Training split IDs
│   ├── val.txt                     # Validation split IDs
│   └── test.txt                    # Test split IDs
├── raw_suim/                       # Raw SUIM data (user provides)
│   ├── train_val/
│   │   ├── images/
│   │   └── masks/
│   └── TEST/
│       ├── images/
│       └── masks/
├── models/
│   ├── unet_resattn.py             # UNet-ResAttn V1
│   ├── unet_resattn_v2.py          # UNet-ResAttn V2 (failed)
│   ├── unet_resattn_v3.py          # UNet-ResAttn V3 (best) 🏆
│   ├── suimnet.py                  # SUIM-Net baseline
│   ├── deeplab_resnet.py           # DeepLabV3 baseline
│   └── blocks/                     # Residual & Attention modules
├── datasets/
│   ├── suim_dataset.py             # Dataset loader
│   └── augmentations.py            # Data augmentation
├── training/
│   ├── train.py                    # Training loop
│   ├── eval.py                     # Evaluation metrics
│   ├── loss.py                     # Dice + CE + Focal loss
│   ├── metrics.py                  # IoU and F-score calculation
│   └── utils.py                    # Checkpoints & utilities
├── checkpoints/                    # Model checkpoints (generated)
│   ├── unet_resattn_v3_best.pth    # Best model (859 MB) 🏆
│   ├── deeplabv3_aug_best.pth      # DeepLabV3 (464 MB)
│   └── ...                         # Other model checkpoints
├── logs/                           # Training logs (generated)
├── organize_suim_dataset.py        # Step 1: Organize raw data
├── create_splits.py                # Step 2: Create train/val/test splits
├── main_train.py                   # Main training script (V1, SUIM-Net, DeepLabV3)
├── train_unet_v2.py                # V2 training script
├── train_unet_v3.py                # V3 training script 🏆
├── train_all_models.sh             # Train all 5 models
├── evaluate.py                     # Individual model evaluation
├── evaluate_with_fscore.py         # Comprehensive evaluation (all models)
├── test_models.py                  # Quick model test
├── requirements.txt                # Python dependencies
├── TRAINING_GUIDE.md               # Training instructions
├── TRAINING_REPORT.md              # Comprehensive analysis
├── FINAL_RESULTS.md                # Results summary
├── STATE_OF_THE_ART.md             # Benchmark analysis
└── README.md                       # This file
```

**Note:** Folders `data/`, `checkpoints/`, `logs/`, and `raw_suim/` are in `.gitignore` due to large file sizes.

---

## Hardware Requirements

### Recommended
- **GPU**: NVIDIA GPU with 8-12GB VRAM (e.g., RTX 3060, RTX 3070, RTX 4070)
- **RAM**: 16GB minimum
- **Storage**: 10GB free space (dataset + checkpoints)
- **OS**: Linux (Ubuntu 20.04+), Windows 10/11, or macOS

### Tested Configuration
- **GPU**: NVIDIA GeForce RTX 3060 (12GB)
- **CUDA**: 12.8
- **Driver**: 570.195.03
- **PyTorch**: 2.9.1+cu128
- **Python**: 3.11.8

### CPU Only
- Training on CPU is **NOT recommended** (10-20× slower)
- Expected time for all models: 150-300 hours vs 15-18 hours on GPU
- For CPU training, reduce batch sizes to 2-4

### Cloud Alternatives
If you don't have a GPU:
- **Google Colab**: Free GPU (limited hours)
- **Kaggle Kernels**: Free GPU (weekly quota)
- **AWS EC2**: g4dn instances (paid)
- **Paperspace**: Gradient instances (paid)

---

## Troubleshooting

### Dataset Setup Issues

**Problem:** `raw_suim/` folder not found
```bash
# Solution: Create it and add your data
mkdir -p raw_suim
# Then manually copy SUIM dataset into raw_suim/
```

**Problem:** "No such file or directory" errors during training
```bash
# Solution: Verify dataset organization
python -c "import os; print('Images:', len(os.listdir('data/images/train')))"
# Should print: Images: 1220

# Re-run organization if needed
python organize_suim_dataset.py
python create_splits.py
```

### Training Issues

**Problem:** CUDA Out of Memory
```bash
# Solution: Reduce batch size
python train_unet_v3.py --epochs 50 --batch_size 4  # instead of 6
python main_train.py --model deeplabv3 --batch_size 2 --epochs 30  # instead of 4
```

**Problem:** "RGB palette error" or "All pixels mapped to class 0"
```bash
# Solution: Verify correct palette in datasets/suim_dataset.py
python -c "from datasets.suim_dataset import RGB_TO_CLASS; print(RGB_TO_CLASS[(255,0,0)])"
# Should print: 1 (not 0)
```

**Problem:** Training too slow
```bash
# Check if GPU is being used
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
# Should print: CUDA: True

# Monitor GPU usage
nvidia-smi
```

### Evaluation Issues

**Problem:** "Checkpoint not found"
```bash
# Check available checkpoints
ls -lh checkpoints/

# Verify checkpoint path
python evaluate.py --model unet_resattn --checkpoint checkpoints/unet_resattn_v3_best.pth
```

**Problem:** Import errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

---

## FAQ

**Q: How long does training take?**  
A: On NVIDIA RTX 3060:
- All 5 models: 15-18 hours total
- UNet-ResAttn-V3 only: ~3.5 hours
- DeepLabV3 only: ~3 hours

**Q: Which model should I use for production?**  
A: 
- **Maximum accuracy**: UNet-ResAttn-V3 (51.91% mIoU)
- **Balanced efficiency**: DeepLabV3 (50.65% mIoU, fewer parameters)
- **Real-time/edge**: SUIM-Net (33.12% mIoU, only 7.76M params)

**Q: Can I use my own underwater images?**  
A: Yes, but you'll need to:
1. Annotate your images with 8 class masks
2. Format masks with the same RGB palette as SUIM
3. Update the dataset loader
4. Fine-tune pre-trained checkpoints on your data

**Q: Why did UNet-ResAttn-V2 perform worse than V1?**  
A: Over-engineering without pre-training. See `TRAINING_REPORT.md` for detailed analysis.

**Q: What's the difference between IoU and F-score?**  
A: 
- **IoU (Intersection over Union)**: Overlap / Union of prediction and ground truth
- **F-score (Dice coefficient)**: 2 × Overlap / (Pred + GT), more lenient than IoU
- Both measure segmentation quality, F-score typically 8-12% higher

**Q: Can I skip training and use pre-trained checkpoints?**  
A: Checkpoints are too large for git (~2.5GB total). You must train locally or download from a shared source.

**Q: How do I cite this work?**  
A:
```
@misc{cv_assign2_2025,
  author = {CV Assignment 2 Team},
  title = {Underwater Semantic Segmentation with UNet-ResAttn-V3},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/ghostcat-pro/CV_Assign2}
}
```

---

## References

### Dataset
- **SUIM Dataset**: Islam, M. J., et al. "Semantic segmentation of underwater imagery: Dataset and benchmark." *IROS 2020*.
- **Dataset URL**: https://irvlab.cs.umn.edu/resources/suim-dataset
- **Paper**: https://arxiv.org/pdf/2004.01241.pdf

### Models & Techniques
- **DeepLabV3**: Chen, L. C., et al. "Rethinking atrous convolution for semantic image segmentation." *arXiv 2017*.
- **UNet**: Ronneberger, O., et al. "U-net: Convolutional networks for biomedical image segmentation." *MICCAI 2015*.
- **Attention Gates**: Oktay, O., et al. "Attention u-net: Learning where to look for the pancreas." *MIDL 2018*.
- **Focal Loss**: Lin, T. Y., et al. "Focal loss for dense object detection." *ICCV 2017*.
- **ResNet**: He, K., et al. "Deep residual learning for image recognition." *CVPR 2016*.

### Frameworks
- **PyTorch**: https://pytorch.org/
- **Albumentations**: https://albumentations.ai/
- **torchvision**: https://pytorch.org/vision/

---

## License

This project is for educational purposes (CV Assignment 2). The SUIM dataset has its own license - please refer to the original authors.

---

## Contact

For questions or issues:
- Open an issue on GitHub: https://github.com/ghostcat-pro/CV_Assign2/issues
- See comprehensive documentation in `TRAINING_REPORT.md`

---

**README Last Updated:** November 25, 2025  
**Best Model:** UNet-ResAttn-V3 (51.91% mIoU, 61.52% F-score) 🏆
