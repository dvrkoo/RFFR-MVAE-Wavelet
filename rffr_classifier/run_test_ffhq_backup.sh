#!/bin/bash
################################################################################
# RFFR FFHQ Testing Script
# 
# Tests RFFR model trained with 3 fake frames on FFHQ benchmark with ALL
# generative models combined (StyleGAN1/2/3/XL, StableDiffusion v1.4/v2.1)
#
# Author: OpenCode
# Date: 2025-12-20
################################################################################

# Configuration
GPU_ID="1"

# Checkpoints - Test both MAE and MAE_VAE models
#
CHECKPOINT_MAE="/andromeda/personal/nmarini/RFFR/rffr_classifier/checkpoint/mae_vae/3branch_wavelet_residual_F2F_All_2025-10-31-15:30:50_85d8e3/best_model/2025-10-31-15:30:50_85d8e3/3__AUC_0.99622_150.pth.tar"
CHECKPOINT_MAE_VAE="/andromeda/personal/nmarini/RFFR/rffr_classifier/checkpoint/mae_vae/3branch_wavelet_residual_F2F_All_Fake3_2025-10-31-09:08:12_2004fc/best_model/2025-10-31-09:08:12_2004fc/3__AUC_0.9948_270.pth.tar"

# Dataset path
FFHQ_ROOT="/oblivion/Datasets/FFHQ"

# Testing parameters
BATCH_SIZE=128
WORKERS=8
SEED=666

# Test on ALL generators combined
GENERATOR="all"

################################################################################

# Change to RFFR classifier directory
cd /andromeda/personal/nmarini/RFFR/rffr_classifier

echo "================================================================================================"
echo "RFFR FFHQ TESTING (ALL Generators Combined)"
echo "================================================================================================"
echo "Start time: $(date)"
echo ""
echo "Configuration:"
echo "  GPU ID: ${GPU_ID}"
echo "  FFHQ Root: ${FFHQ_ROOT}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Workers: ${WORKERS}"
echo "  Random Seed: ${SEED}"
echo "  Generator Mode: ${GENERATOR} (10K real + 60K fake = 70K total)"
echo "================================================================================================"
echo ""

# Test MAE model
echo "================================================================================================"
echo "Testing RFFR MAE Model (Fake3)"
echo "================================================================================================"
echo "Checkpoint: ${CHECKPOINT_MAE}"
echo ""

CUDA_VISIBLE_DEVICES=${GPU_ID} python3 test_ffhq.py \
    --checkpoint "${CHECKPOINT_MAE}" \
    --ffhq_root "${FFHQ_ROOT}" \
    --batch_size ${BATCH_SIZE} \
    --workers ${WORKERS} \
    --seed ${SEED} \
    --gpu ${GPU_ID} \
    --generators ${GENERATOR}

MAE_EXIT_CODE=$?

echo ""
echo "MAE model testing completed with exit code: ${MAE_EXIT_CODE}"
echo ""

# Test MAE_VAE model
echo "================================================================================================"
echo "Testing RFFR MAE_VAE Model (Fake3)"
echo "================================================================================================"
echo "Checkpoint: ${CHECKPOINT_MAE_VAE}"
echo ""

CUDA_VISIBLE_DEVICES=${GPU_ID} python3 test_ffhq.py \
    --checkpoint "${CHECKPOINT_MAE_VAE}" \
    --ffhq_root "${FFHQ_ROOT}" \
    --batch_size ${BATCH_SIZE} \
    --workers ${WORKERS} \
    --seed ${SEED} \
    --gpu ${GPU_ID} \
    --generators ${GENERATOR}

MAE_VAE_EXIT_CODE=$?

echo ""
echo "MAE_VAE model testing completed with exit code: ${MAE_VAE_EXIT_CODE}"
echo ""

echo "================================================================================================"
echo "All testing completed!"
echo "End time: $(date)"
echo "================================================================================================"

if [ ${MAE_EXIT_CODE} -eq 0 ] && [ ${MAE_VAE_EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "✓ All tests completed successfully!"
    echo ""
    echo "Results saved to:"
    echo "  MAE: $(dirname ${CHECKPOINT_MAE})/test_ffhq_*.txt"
    echo "  MAE_VAE: $(dirname ${CHECKPOINT_MAE_VAE})/test_ffhq_*.txt"
else
    echo ""
    echo "✗ Some tests failed!"
    echo "  MAE exit code: ${MAE_EXIT_CODE}"
    echo "  MAE_VAE exit code: ${MAE_VAE_EXIT_CODE}"
fi

exit $(( ${MAE_EXIT_CODE} + ${MAE_VAE_EXIT_CODE} ))
