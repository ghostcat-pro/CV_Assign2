#!/bin/bash
# Final training script for remaining models
# Keras models: CPU only, 10 epochs (CUDA compilation issues)
# PyTorch models: GPU, 50 epochs
#
# Models to train:
#   1. SUIM-Net Keras RSB (with augmentation) - 10 epochs, CPU
#   2. DeepLabV3 (no augmentation) - 50 epochs, GPU
#   3. UNet-ResAttn-V4 (no augmentation) - 50 epochs, GPU
#   4. UWSegFormer (no augmentation) - 50 epochs, GPU
#   5. SUIM-Net Keras VGG (no augmentation) - 10 epochs, CPU
#   6. SUIM-Net Keras RSB (no augmentation) - 10 epochs, CPU

set -e  # Exit on error

echo "================================================================================"
echo "FINAL TRAINING BATCH - Mixed Models"
echo "================================================================================"
echo "Keras models (CPU, 10 epochs each):"
echo "  1. SUIM-Net Keras (RSB backbone, WITH augmentation)"
echo "  5. SUIM-Net Keras (VGG backbone, NO augmentation)"
echo "  6. SUIM-Net Keras (RSB backbone, NO augmentation)"
echo ""
echo "PyTorch models (GPU, 50 epochs each):"
echo "  2. DeepLabV3 (NO augmentation)"
echo "  3. UNet-ResAttn-V4 (NO augmentation)"
echo "  4. UWSegFormer (NO augmentation)"
echo ""
echo "Estimated time:"
echo "  - Keras models: ~5-7 hours each (CPU, 10 epochs)"
echo "  - PyTorch models: ~12-15 hours total (GPU)"
echo "================================================================================"
echo ""

# Activate virtual environment
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Create log directory
mkdir -p logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="logs/train_final_batch_${TIMESTAMP}.log"
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

# Train 1: SUIM-Net Keras RSB (with augmentation) - CPU, 10 epochs
echo "================================================================================"
echo "Starting Keras model 1/3: RSB with augmentation (CPU, 10 epochs)"
echo "================================================================================"
CUDA_VISIBLE_DEVICES='' train_model \
    "SUIM-Net Keras (RSB, with aug)" \
    "main_train.py" \
    --model suimnet_keras \
    --backbone RSB \
    --epochs 10 \
    --batch_size 8 \
    --augment \
    --lr 1e-4

# Train 2: DeepLabV3 (no augmentation) - GPU, 50 epochs
echo "================================================================================"
echo "Starting PyTorch model 1/3: DeepLabV3 no augmentation (GPU, 50 epochs)"
echo "================================================================================"
train_model \
    "DeepLabV3 (no aug)" \
    "main_train.py" \
    --model deeplabv3 \
    --epochs 50 \
    --batch_size 8 \
    --no-augment \
    --lr 1e-4

# Train 3: UNet-ResAttn-V4 (no augmentation) - GPU, 50 epochs
echo "================================================================================"
echo "Starting PyTorch model 2/3: UNet-ResAttn-V4 no augmentation (GPU, 50 epochs)"
echo "================================================================================"
train_model \
    "UNet-ResAttn-V4 (no aug)" \
    "train_unet_v4.py" \
    --epochs 50 \
    --batch_size 6 \
    --no-augment \
    --lr 1e-4

# Train 4: UWSegFormer (no augmentation) - GPU, 50 epochs
echo "================================================================================"
echo "Starting PyTorch model 3/3: UWSegFormer no augmentation (GPU, 50 epochs)"
echo "================================================================================"
train_model \
    "UWSegFormer (no aug)" \
    "main_train.py" \
    --model uwsegformer \
    --backbone resnet50 \
    --epochs 50 \
    --batch_size 4 \
    --no-augment \
    --lr 6e-5

# Train 5: SUIM-Net Keras VGG (no augmentation) - CPU, 10 epochs
echo "================================================================================"
echo "Starting Keras model 2/3: VGG no augmentation (CPU, 10 epochs)"
echo "================================================================================"
CUDA_VISIBLE_DEVICES='' train_model \
    "SUIM-Net Keras (VGG, no aug)" \
    "main_train.py" \
    --model suimnet_keras \
    --backbone VGG \
    --epochs 10 \
    --batch_size 8 \
    --no-augment \
    --lr 1e-4

# Train 6: SUIM-Net Keras RSB (no augmentation) - CPU, 10 epochs
echo "================================================================================"
echo "Starting Keras model 3/3: RSB no augmentation (CPU, 10 epochs)"
echo "================================================================================"
CUDA_VISIBLE_DEVICES='' train_model \
    "SUIM-Net Keras (RSB, no aug)" \
    "main_train.py" \
    --model suimnet_keras \
    --backbone RSB \
    --epochs 10 \
    --batch_size 8 \
    --no-augment \
    --lr 1e-4

echo "================================================================================"
echo "ALL TRAINING COMPLETE!"
echo "================================================================================"
echo "Summary:"
echo ""
echo "Keras Models (10 epochs each, CPU):"
echo "  ✓ SUIM-Net Keras (RSB, with aug) - checkpoints/suimnet_keras_rsb_8cls_aug_best.weights.h5"
echo "  ✓ SUIM-Net Keras (VGG, no aug) - checkpoints/suimnet_keras_vgg_8cls_noaug_best.weights.h5"
echo "  ✓ SUIM-Net Keras (RSB, no aug) - checkpoints/suimnet_keras_rsb_8cls_noaug_best.weights.h5"
echo ""
echo "PyTorch Models (50 epochs each, GPU):"
echo "  ✓ DeepLabV3 (no aug) - checkpoints/deeplabv3_8cls_noaug_best.pth"
echo "  ✓ UNet-ResAttn-V4 (no aug) - checkpoints/unet_resattn_v4_8cls_noaug_best.pth"
echo "  ✓ UWSegFormer (no aug) - checkpoints/uwsegformer_8cls_noaug_best.pth"
echo ""
echo "Full log: $LOGFILE"
echo ""
echo "Next steps:"
echo "  1. Run comprehensive evaluation on all models"
echo "  2. Compare augmented vs non-augmented results"
echo "  3. Update reports in markdowns/"
echo "================================================================================"
