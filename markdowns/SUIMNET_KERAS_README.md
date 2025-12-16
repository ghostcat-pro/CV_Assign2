# SUIM-Net Keras Training Guide

This directory contains the original Keras/TensorFlow implementation of SUIM-Net from the paper:
**"Semantic Segmentation of Underwater Imagery: Dataset and Benchmark"** (https://arxiv.org/pdf/2004.01241.pdf)

## Overview

SUIM-Net is designed specifically for underwater image segmentation with two backbone options:
- **RSB (Residual Skip Blocks)**: Custom lightweight encoder (320x240 input)
- **VGG16**: Pre-trained VGG16 encoder (320x256 input)

## Installation

Install the required Keras/TensorFlow dependencies:

```bash
pip install tensorflow==2.10.0 keras==2.10.0
```

Or install all dependencies:

```bash
pip install -r requirements.txt
```

## Data Preparation

Your data directory should be organized as:

```
data/
├── images/
│   ├── *.jpg  (training images)
├── masks/
│   ├── *.bmp  (RGB masks with SUIM color coding)
```

The script uses Keras `ImageDataGenerator` which expects this structure.

## Training

### Basic Training (VGG backbone, 5 classes - paper default)

```bash
python train_suimnet_keras.py --base VGG --epochs 50 --batch_size 8
```

### RSB Backbone Training

```bash
python train_suimnet_keras.py --base RSB --epochs 50 --batch_size 8
```

### Training with 8 Classes (full SUIM classes)

```bash
python train_suimnet_keras.py --base VGG --epochs 50 --batch_size 8 --n_classes 8
```

### Resume Training from Checkpoint

```bash
python train_suimnet_keras.py --base VGG --epochs 50 --resume checkpoints/suimnet_vgg_5cls.hdf5
```

### Advanced Options

```bash
python train_suimnet_keras.py \
    --base VGG \
    --epochs 50 \
    --batch_size 8 \
    --n_classes 5 \
    --train_dir data \
    --ckpt_dir checkpoints \
    --steps_per_epoch 200 \
    --early_stopping \
    --tensorboard \
    --verbose
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--train_dir` | str | `data` | Path to training data directory |
| `--base` | str | `VGG` | Backbone: `RSB` or `VGG` |
| `--n_classes` | int | `5` | Number of output classes (5 or 8) |
| `--epochs` | int | `50` | Number of training epochs |
| `--batch_size` | int | `8` | Batch size |
| `--steps_per_epoch` | int | auto | Steps per epoch (auto-calculated if None) |
| `--ckpt_dir` | str | `checkpoints` | Checkpoint directory |
| `--resume` | str | None | Path to checkpoint to resume from |
| `--early_stopping` | flag | False | Enable early stopping |
| `--tensorboard` | flag | False | Enable TensorBoard logging |
| `--verbose` | flag | False | Print model summary |

## Class Mapping

### 5 Classes (Paper Default - Merged)
Following the paper's class merging strategy:

0. **Background + Sea-floor** (Black + White)
1. **Human divers** (Blue)
2. **Aquatic plants** (Green)
3. **Wrecks** (Cyan)
4. **Other objects** (Robots + Reefs + Fish: Red + Magenta + Yellow)

### 8 Classes (Full SUIM)
All original SUIM classes:

0. **Background** (Black: 0,0,0)
1. **Human divers** (Blue: 0,0,255)
2. **Aquatic plants** (Green: 0,255,0)
3. **Wrecks** (Cyan: 0,255,255)
4. **Robots** (Red: 255,0,0)
5. **Reefs** (Magenta: 255,0,255)
6. **Sea-floor** (White: 255,255,255)
7. **Fish** (Yellow: 255,255,0)

## Data Augmentation

The training pipeline includes the following augmentations (from the paper):
- Rotation (±20%)
- Width/Height shift (±5%)
- Shear (±5%)
- Zoom (±5%)
- Horizontal flip
- Nearest-neighbor fill mode

## Model Checkpoints

Checkpoints are saved in the format:
- `suimnet_rsb_5cls.hdf5` - RSB backbone, 5 classes
- `suimnet_vgg_5cls.hdf5` - VGG backbone, 5 classes
- `suimnet_vgg_8cls.hdf5` - VGG backbone, 8 classes

The best model (based on training loss) is saved during training.
A final model snapshot is also saved at the end: `*_final.hdf5`

## Model Architecture

### RSB Backbone
- Custom residual skip blocks
- Input: 320x240x3
- Lightweight encoder-decoder
- ~8M parameters

### VGG16 Backbone
- Pre-trained VGG16 encoder
- Input: 320x256x3
- Decoder with skip connections
- ~14M parameters (VGG pre-trained on ImageNet)

## TensorBoard Monitoring

To enable TensorBoard logging:

```bash
python train_suimnet_keras.py --base VGG --tensorboard
```

View logs:

```bash
tensorboard --logdir checkpoints/logs
```

## Performance Notes

### Memory Requirements
- VGG16: ~2-3 GB GPU memory (batch_size=8)
- RSB: ~1-2 GB GPU memory (batch_size=8)

### Training Time
- ~2-3 minutes/epoch on modern GPU (1500 images)
- ~15-20 minutes/epoch on CPU

### Recommended Settings
- **Quick training**: `--epochs 30 --batch_size 8`
- **Best results**: `--epochs 50 --batch_size 8 --early_stopping`
- **Limited memory**: `--batch_size 4` or use RSB backbone

## Differences from PyTorch Models

This Keras implementation follows the **original paper** exactly:
- Uses Keras `ImageDataGenerator` for data loading
- Binary cross-entropy loss
- Adam optimizer with learning rate 1e-4
- Different architecture than PyTorch UNet variants

For PyTorch training, use:
- `main_train.py` - UNet-ResAttn variants
- `train_unet_v*.py` - Specific UNet versions

## Troubleshooting

### TensorFlow/Keras Import Error
```bash
pip install tensorflow==2.10.0 keras==2.10.0
```

### CUDA/cuDNN Issues
Use CPU-only TensorFlow if GPU issues occur:
```bash
pip install tensorflow-cpu==2.10.0
```

### Model Input/Output Mismatch
Ensure your masks are RGB format with SUIM color coding. The script automatically converts RGB masks to class indices.

### Steps Per Epoch Too High
Reduce `--steps_per_epoch` to match your dataset size:
```bash
python train_suimnet_keras.py --steps_per_epoch 100
```

## Citation

If you use SUIM-Net, please cite the paper:

```bibtex
@inproceedings{islam2020suim,
  title={Semantic Segmentation of Underwater Imagery: Dataset and Benchmark},
  author={Islam, Md Jahidul and Edge, Chelsey and Xiao, Yuyang and Luo, Peigen and Mehtaz, Muntaqim and Morse, Christopher and Enan, Sadman Sakib and Sattar, Junaed},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2020}
}
```

## Related Files

- `models/paper/suimnet.py` - SUIM-Net model implementation
- `data/utils/keras_data_utils.py` - Keras data generators
- `main_train.py` - PyTorch training script (different models)
- `evaluate.py` - Model evaluation script
