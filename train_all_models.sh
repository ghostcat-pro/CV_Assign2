#!/bin/bash
# Script to train all 3 models sequentially
# Run with: bash train_all_models.sh

# Activate virtual environment
source venv/bin/activate

# Create a log directory
mkdir -p logs

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
        --augment \
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

# Train all models with augmentation
echo "Training with Data Augmentation" | tee -a "$LOG_FILE"
echo "================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# 1. SUIM-Net (lightweight, fast)
train_model "suimnet" 50 8

# 2. UNet-ResAttn (custom architecture)
train_model "unet_resattn" 50 8

# 3. DeepLabV3 (larger model, needs smaller batch)
train_model "deeplabv3" 30 4

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
