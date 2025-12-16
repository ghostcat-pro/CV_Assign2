#!/bin/bash
# Script to train all models sequentially
# Configured to run with MERGED CLASSES (5 classes) and WITHOUT augmentation
# Run with: bash train_all_models.sh
# Override with full augmentation: bash train_all_models.sh --augment
# Override with 8 classes: bash train_all_models.sh --no-merge-classes

# Activate virtual environment (Windows path)
source venv/Scripts/activate

# Create a log directory
mkdir -p logs

# Default: no augmentation, merged classes (5 classes)
AUG_FLAG="--no-augment"
MERGE_FLAG="--merge-classes"

# Parse command line arguments to override defaults
for arg in "$@"; do
    if [ "$arg" = "--augment" ]; then
        AUG_FLAG=""
        echo "Override: Running WITH data augmentation"
    elif [ "$arg" = "--no-merge-classes" ]; then
        MERGE_FLAG=""
        echo "Override: Running with ORIGINAL classes (8 classes)"
    fi
done

if [ -z "$AUG_FLAG" ]; then
    echo "Running WITH data augmentation"
else
    echo "Running WITHOUT data augmentation (DEFAULT)"
fi
if [ -z "$MERGE_FLAG" ]; then
    echo "Running with ORIGINAL classes (8 classes)"
else
    echo "Running with MERGED classes (5 classes, DEFAULT)"
fi

# Get current timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/train_all_${TIMESTAMP}.log"

echo "========================================" | tee -a "$LOG_FILE"
echo "Starting Sequential Training of All Models" | tee -a "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Function to train a model
train_model() {
    local model_name=$1
    local epochs=$2
    local batch_size=$3
    local backbone=$4  # Optional backbone parameter
    
    local display_name="$model_name"
    if [ -n "$backbone" ]; then
        display_name="$model_name (backbone: $backbone)"
    fi
    
    echo "========================================" | tee -a "$LOG_FILE"
    echo "Training: $display_name" | tee -a "$LOG_FILE"
    echo "Epochs: $epochs | Batch Size: $batch_size" | tee -a "$LOG_FILE"
    echo "Started at: $(date)" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    
    # Build command with optional backbone
    if [ -n "$backbone" ]; then
        python main_train.py \
            --model "$model_name" \
            --backbone "$backbone" \
            --epochs "$epochs" \
            --batch_size "$batch_size" \
            $AUG_FLAG \
            $MERGE_FLAG \
            2>&1 | tee -a "$LOG_FILE"
    else
        python main_train.py \
            --model "$model_name" \
            --epochs "$epochs" \
            --batch_size "$batch_size" \
            $AUG_FLAG \
            $MERGE_FLAG \
            2>&1 | tee -a "$LOG_FILE"
    fi
    
    exit_code=${PIPESTATUS[0]}
    
    if [ $exit_code -eq 0 ]; then
        echo "" | tee -a "$LOG_FILE"
        echo "✓ $display_name training completed successfully!" | tee -a "$LOG_FILE"
        echo "Completed at: $(date)" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
    else
        echo "" | tee -a "$LOG_FILE"
        echo "✗ $display_name training failed with exit code $exit_code" | tee -a "$LOG_FILE"
        echo "Failed at: $(date)" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
    fi
    
    # Clear GPU memory between models
    sleep 5
}

# Display augmentation status
if [ -z "$AUG_FLAG" ]; then
    echo "Training Mode: WITH Data Augmentation" | tee -a "$LOG_FILE"
else
    echo "Training Mode: WITHOUT Data Augmentation" | tee -a "$LOG_FILE"
fi
echo "================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# 1. UWSegFormer V2 (enhanced with color restoration, multi-head attention - reduced batch size for stability)
# train_model "uwsegformer_v2" 50 4

# 2. Keras SUIM-Net with RSB backbone
train_model "suimnet_keras" 50 8 "RSB"

# 3. Keras SUIM-Net with VGG backbone
train_model "suimnet_keras" 50 8 "VGG"

# 4. DeepLabV3 (larger model, needs smaller batch)
train_model "deeplabv3" 30 4

# 5. UNet-ResAttn V4 (with pre-trained ResNet-50, edge detection, deep supervision)
train_model "unet_resattn_v4" 50 6

# Final summary
echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "All Training Jobs Completed!" | tee -a "$LOG_FILE"
echo "Finished at: $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# List all saved checkpoints
echo "Saved Checkpoints:" | tee -a "$LOG_FILE"
ls -lh checkpoints/*.pth 2>/dev/null | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "Log file saved to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "Done!" | tee -a "$LOG_FILE"
