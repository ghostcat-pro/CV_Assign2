#!/bin/bash
# Training script for models WITHOUT augmentation (8 classes)
# Models to train: SUIM-Net, DeepLabV3, UNet-ResAttn-V4, UWSegFormer

set -e  # Exit on error

echo "================================================================================"
echo "TRAINING MODELS WITHOUT AUGMENTATION - 8 Classes"
echo "================================================================================"
echo "Models to train:"
echo "  1. SUIM-Net (PyTorch)"
echo "  2. DeepLabV3-ResNet50"
echo "  3. UNet-ResAttn-V4 (ASPP + CBAM)"
echo "  4. UWSegFormer (ResNet-50)"
echo ""
echo "Training Configuration:"
echo "  - Classes: 8 (original SUIM)"
echo "  - Augmentation: DISABLED"
echo "  - Epochs: 50 for all models"
echo ""
echo "Estimated time: 12-15 hours total"
echo "================================================================================"
echo ""

# Activate virtual environment if exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Create log directory if it doesn't exist
mkdir -p logs

# Timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="logs/train_noaug_${TIMESTAMP}.log"

echo "Logging to: $LOGFILE"
echo ""

# Function to train a model
train_model() {
    local model_name=$1
    local script=$2
    shift 2
    local args="$@"
    
    echo "================================================================================"
    echo "Training: $model_name (NO AUGMENTATION)"
    echo "Script: $script"
    echo "Args: $args"
    echo "Start time: $(date)"
    echo "================================================================================"
    
    if python $script $args 2>&1 | tee -a "$LOGFILE"; then
        echo ""
        echo "✓ $model_name training completed successfully!"
        echo "Completion time: $(date)"
    else
        echo ""
        echo "✗ $model_name training failed!"
        echo "Check logs for details: $LOGFILE"
        return 1
    fi
    
    echo ""
    echo "Pausing for 10 seconds before next model..."
    sleep 10
    echo ""
}

# Train SUIM-Net (no augmentation)
train_model \
    "SUIM-Net" \
    "main_train.py" \
    --model suimnet \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-4

# Train DeepLabV3 (no augmentation)
train_model \
    "DeepLabV3-ResNet50" \
    "main_train.py" \
    --model deeplabv3 \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-4

# Train UNet-ResAttn-V4 (no augmentation)
train_model \
    "UNet-ResAttn-V4" \
    "train_unet_v4.py" \
    --epochs 50 \
    --batch_size 6 \
    --lr 1e-4

# Train UWSegFormer (no augmentation)
train_model \
    "UWSegFormer" \
    "main_train.py" \
    --model uwsegformer \
    --backbone resnet50 \
    --epochs 50 \
    --batch_size 4 \
    --lr 6e-5

echo "================================================================================"
echo "ALL TRAINING COMPLETE!"
echo "================================================================================"
echo "Summary:"
echo "  ✓ SUIM-Net (no augmentation)"
echo "  ✓ DeepLabV3 (no augmentation)"
echo "  ✓ UNet-ResAttn-V4 (no augmentation)"
echo "  ✓ UWSegFormer (no augmentation)"
echo ""
echo "Checkpoints saved in: checkpoints/"
echo "  - suimnet_8cls_noaug_best.pth"
echo "  - deeplabv3_8cls_noaug_best.pth"
echo "  - unet_resattn_v4_8cls_noaug_best.pth"
echo "  - uwsegformer_8cls_noaug_best.pth"
echo ""
echo "Full log: $LOGFILE"
echo ""
echo "Next steps:"
echo "  1. Run comprehensive evaluation on all models"
echo "  2. Compare augmented vs non-augmented results"
echo "  3. Update reports in markdowns/"
echo "================================================================================"
