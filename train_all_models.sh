#!/bin/bash
# Script to train all 3 models sequentially
# Run with: bash train_all_models.sh
# Run without augmentation: bash train_all_models.sh --no-augment
# Run with merged classes (6 classes): bash train_all_models.sh --merge-classes
# Combine flags: bash train_all_models.sh --no-augment --merge-classes

# Activate virtual environment (Windows path)
source venv/Scripts/activate

# Create a log directory
mkdir -p logs

# Parse command line arguments
AUG_FLAG=""
MERGE_FLAG=""
for arg in "$@"; do
    if [ "$arg" = "--no-augment" ]; then
        AUG_FLAG="--no-augment"
        echo "Running WITHOUT data augmentation"
    elif [ "$arg" = "--merge-classes" ]; then
        MERGE_FLAG="--merge-classes"
        echo "Running with MERGED classes (6 classes instead of 8)"
    fi
done

if [ -z "$AUG_FLAG" ]; then
    echo "Running WITH data augmentation (default)"
fi
if [ -z "$MERGE_FLAG" ]; then
    echo "Running with ORIGINAL classes (8 classes)"
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
    
    echo "========================================" | tee -a "$LOG_FILE"
    echo "Training: $model_name" | tee -a "$LOG_FILE"
    echo "Epochs: $epochs | Batch Size: $batch_size" | tee -a "$LOG_FILE"
    echo "Started at: $(date)" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    
    python main_train.py \
        --model "$model_name" \
        --epochs "$epochs" \
        --batch_size "$batch_size" \
        $AUG_FLAG \
        $MERGE_FLAG \
        2>&1 | tee -a "$LOG_FILE"
    
    exit_code=${PIPESTATUS[0]}
    
    if [ $exit_code -eq 0 ]; then
        echo "" | tee -a "$LOG_FILE"
        echo "✓ $model_name training completed successfully!" | tee -a "$LOG_FILE"
        echo "Completed at: $(date)" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
    else
        echo "" | tee -a "$LOG_FILE"
        echo "✗ $model_name training failed with exit code $exit_code" | tee -a "$LOG_FILE"
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

# 1. SUIM-Net (lightweight, fast)
#train_model "suimnet" 50 8

# 2. UNet-ResAttn V1 (baseline custom architecture)
#train_model "unet_resattn" 50 8

# 3. UNet-ResAttn V2 (with SE blocks, SPP, deep supervision)
train_model "unet_resattn_v2" 60 8

# 4. UNet-ResAttn V3 (with pre-trained ResNet-50, focal loss, 384x384)
train_model "unet_resattn_v3" 50 6

# 5. DeepLabV3 (larger model, needs smaller batch)
train_model "deeplabv3" 30 4

# 6. UWSegFormer with ResNet-50 backbone (default)
train_model "uwsegformer" 50 8

# Optional: Train UWSegFormer with MIT-B0 backbone
# Uncomment the line below to also train with transformer backbone
# train_model "uwsegformer --backbone mit_b0" 50 8

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
