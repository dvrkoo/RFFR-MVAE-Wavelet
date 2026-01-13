#!/bin/bash

# Test the latest FFHQ MAE-VAE Wavelet model (Epoch 1025)
# This is the best checkpoint from FFHQ Stage 1 training

GPU_ID=${1:-1}  # Default GPU 1, can override with first argument

CHECKPOINT="/seidenas/users/nmarini/generative_checkpoint/mae_vae/FFHQ_mae_vae_STAGE1_best/best_loss_0.01231_1025.pth.tar"
FFHQ_ROOT="/oblivion/Datasets/FFHQ"

cd /andromeda/personal/nmarini/RFFR/rffr_classifier

echo "=============================================================================="
echo "Testing FFHQ MAE-VAE Wavelet Model (Epoch 1025, Loss 0.01231)"
echo "=============================================================================="
echo "Checkpoint: $CHECKPOINT"
echo "GPU: $GPU_ID"
echo "FFHQ Root: $FFHQ_ROOT"
echo "Start time: $(date)"
echo "=============================================================================="

# Test on all FFHQ generators
CUDA_VISIBLE_DEVICES=${GPU_ID} python3 test_ffhq.py \
    --checkpoint "${CHECKPOINT}" \
    --ffhq_root "${FFHQ_ROOT}" \
    --batch_size 128 \
    --workers 8 \
    --seed 666 \
    --gpu ${GPU_ID} \
    --generators all stylegan1-psi-0.5 stylegan2-psi-0.5 stylegan3-psi-0.5 styleganxl-psi-0.5 sdv1_4 sdv2_1

EXIT_CODE=$?

echo ""
echo "=============================================================================="
echo "Testing completed!"
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=============================================================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ Testing successful!"
    echo ""
    echo "Results should be saved in the checkpoint directory or test_results/"
else
    echo ""
    echo "✗ Testing failed with exit code: $EXIT_CODE"
fi

exit $EXIT_CODE
