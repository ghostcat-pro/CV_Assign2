#!/bin/bash
# Training script for the 3 remaining models (8 classes, with augmentation)
# Models to train: UNet-ResAttn-V4, UWSegFormer, UWSegFormer-V2

set -e  # Exit on error

echo "================================================================================"
echo "TRAINING REMAINING MODELS - 8 Classes with Augmentation"
echo "================================================================================"
echo "Models to train:"
echo "  1. UNet-ResAttn-V4 (ASPP + CBAM + Underwater Color Correction)"
echo "  2. UWSegFormer (ResNet-50 + UIQA + MAA Decoder)"
echo ""
echo "NOTE: UWSegFormer-V2 is not implemented yet (file missing)"
echo ""
echo "Training Configuration:"
echo "  - Classes: 8 (original SUIM)"
echo "  - Augmentation: Enabled"
echo "  - Epochs: 50 (both models)"
echo "  - Batch size: 6 (V4), 4 (UWSegFormer)"
echo ""
echo "Estimated time: 8-10 hours total"
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
LOGFILE="logs/train_remaining_${TIMESTAMP}.log"

echo "Logging to: $LOGFILE"
echo ""

# Function to train a model
train_model() {
    local model_name=$1
    local script=$2
    shift 2
    local args="$@"
    
    echo "================================================================================"
    echo "Training: $model_name"
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

# Train UNet-ResAttn-V4
train_model \
    "UNet-ResAttn-V4" \
    "train_unet_v4.py" \
    --epochs 50 \
    --batch_size 6 \
    --augment \
    --lr 1e-4

# Train UWSegFormer (ResNet-50 backbone)
train_model \
    "UWSegFormer" \
    "main_train.py" \
    --model uwsegformer \
    --backbone resnet50 \
    --epochs 50 \
    --batch_size 4 \
    --augment \
    --lr 6e-5

echo "================================================================================"
echo "ALL TRAINING COMPLETE!"
echo "================================================================================"
echo "Summary:"
echo "  ✓ UNet-ResAttn-V4"
echo "  ✓ UWSegFormer"
echo "  ✗ UWSegFormer-V2 (not implemented - file missing)"
echo ""
echo "Checkpoints saved in: checkpoints/"
echo "  - unet_resattn_v4_8cls_aug_best.pth"
echo "  - uwsegformer_8cls_aug_best.pth"
echo ""
echo "Full log: $LOGFILE"
echo ""
echo "Next steps:"
echo "  1. Run evaluation on 7 models (5 existing + 2 new)"
echo "  2. Review results and update reports"
echo "================================================================================"
