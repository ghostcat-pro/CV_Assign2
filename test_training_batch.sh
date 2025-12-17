#!/bin/bash
# TEST - 2 Keras models (VGG and RSB no augmentation)

set -e

echo "================================================================================"
echo "TEST RUN - 2 KERAS MODELS (NO AUGMENTATION)"
echo "================================================================================"
echo "Testing:  1. SUIM-Net Keras (VGG, NO aug)  2. SUIM-Net Keras (RSB, NO aug)"
echo "================================================================================"

if [ -d "venv" ]; then
    source venv/bin/activate
fi

mkdir -p logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="logs/test_batch_${TIMESTAMP}.log"
echo "Logging to: $LOGFILE"

test_model() {
    echo "================================================================================"
    echo "Testing: $1"
    echo "Start: $(date)"
    echo "================================================================================"
    shift
    if python "$@" 2>&1 | tee -a "$LOGFILE"; then
        echo "✓ Test completed!"
    else
        echo "✗ Test failed!"
        return 1
    fi
    sleep 10
}

echo "Test 1: VGG no aug (CPU)"
CUDA_VISIBLE_DEVICES='' test_model "SUIM-Net Keras (VGG, no aug)" main_train.py --model suimnet_keras --backbone VGG --epochs 1 --batch_size 8 --no-augment --lr 1e-4

echo "Test 2: RSB no aug (CPU)"
CUDA_VISIBLE_DEVICES='' test_model "SUIM-Net Keras (RSB, no aug)" main_train.py --model suimnet_keras --backbone RSB --epochs 1 --batch_size 8 --no-augment --lr 1e-4

echo "================================================================================"
echo "✓ BOTH TESTS COMPLETE! You can now run: ./train_final_batch.sh"
echo "================================================================================"
