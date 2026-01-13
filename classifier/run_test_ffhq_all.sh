#!/bin/bash
################################################################################
# RFFR FFHQ Testing Script - ALL + Individual Generators
# 
# Tests RFFR models on FFHQ with:
#   - All generators combined (70K total)
#   - Each individual generator (20K each)
#
# Matches RECCE test protocol with 7 datasets:
#   1. all (StyleGAN1/2/3/XL + SD v1.4/v2.1)
#   2. stylegan1-psi-0.5
#   3. stylegan2-psi-0.5
#   4. stylegan3-psi-0.5
#   5. styleganxl-psi-0.5
#   6. sdv1_4
#   7. sdv2_1
#
# Author: OpenCode
# Date: 2025-12-20
################################################################################

# Configuration
GPU_ID=${1:-"1"}  # GPU ID, default 1

# Checkpoints - Test both MAE and MAE_VAE models
# CHECKPOINT_MAE="/andromeda/personal/nmarini/RFFR/rffr_classifier/checkpoint/mae_vae/3branch_wavelet_residual_F2F_All_2025-10-31-15:30:50_85d8e3/best_model/2025-10-31-15:30:50_85d8e3/3__AUC_0.99622_150.pth.tar"
# CHECKPOINT_MAE_VAE="/andromeda/personal/nmarini/RFFR/rffr_classifier/checkpoint/mae_vae/3branch_wavelet_residual_F2F_All_Fake3_2025-10-31-09:08:12_2004fc/best_model/2025-10-31-09:08:12_2004fc/3__AUC_0.9948_270.pth.tar"
#
#
CHECKPOINT_MAE="/andromeda/personal/nmarini/RFFR/rffr_classifier/checkpoint/mae/2branch_standard_F2F_All_2025-11-06-19:24:15_0bb195/best_model/2025-11-06-19:24:15_0bb195/1__AUC_0.80152_115.pth.tar"

# CHECKPOINT_MAE="/andromeda/personal/nmarini/RFFR/rffr_classifier/checkpoint/mae/2branch_standard_F2F_All_Fake3_2025-10-27-17:49:08_85bcc4/1__AUC_0.73587_150.pth.tar"

# Dataset path
FFHQ_ROOT="/oblivion/Datasets/FFHQ"

# Testing parameters
BATCH_SIZE=128
WORKERS=8
SEED=666

# All generators to test (7 datasets total)
ALL_GENERATORS="all stylegan1-psi-0.5 stylegan2-psi-0.5 stylegan3-psi-0.5 styleganxl-psi-0.5 sdv1_4 sdv2_1"

################################################################################

# Change to RFFR classifier directory
cd /andromeda/personal/nmarini/RFFR/rffr_classifier || exit 1

echo "================================================================================================"
echo "RFFR FFHQ TESTING - ALL + Individual Generators (Like RECCE)"
echo "================================================================================================"
echo "Start time: $(date)"
echo ""
echo "Configuration:"
echo "  GPU ID: ${GPU_ID}"
echo "  FFHQ Root: ${FFHQ_ROOT}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Workers: ${WORKERS}"
echo "  Random Seed: ${SEED}"
echo ""
echo "Generators to test:"
echo "  1. all (70K: 10K real + 60K fake from all 6 generators)"
echo "  2. stylegan1-psi-0.5 (20K: 10K real + 10K fake)"
echo "  3. stylegan2-psi-0.5 (20K: 10K real + 10K fake)"
echo "  4. stylegan3-psi-0.5 (20K: 10K real + 10K fake)"
echo "  5. styleganxl-psi-0.5 (20K: 10K real + 10K fake)"
echo "  6. sdv1_4 (20K: 10K real + 10K fake)"
echo "  7. sdv2_1 (20K: 10K real + 10K fake)"
echo "================================================================================================"
echo ""

# Function to test a checkpoint
test_checkpoint() {
    local checkpoint=$1
    local checkpoint_name=$2
    
    echo ""
    echo "================================================================================================"
    echo "Testing ${checkpoint_name}"
    echo "================================================================================================"
    echo "Checkpoint: ${checkpoint}"
    echo ""
    
    CUDA_VISIBLE_DEVICES=${GPU_ID} python3 test_ffhq.py \
        --checkpoint "${checkpoint}" \
        --ffhq_root "${FFHQ_ROOT}" \
        --batch_size ${BATCH_SIZE} \
        --workers ${WORKERS} \
        --seed ${SEED} \
        --gpu ${GPU_ID} \
        --generators ${ALL_GENERATORS}
    
    local exit_code=$?
    
    echo ""
    if [ ${exit_code} -eq 0 ]; then
        echo "✓ ${checkpoint_name} testing completed successfully!"
    else
        echo "✗ ${checkpoint_name} testing failed with exit code: ${exit_code}"
    fi
    echo ""
    
    return ${exit_code}
}

# Test MAE model
test_checkpoint "${CHECKPOINT_MAE}" "RFFR MAE Model (All Frames)"
MAE_EXIT_CODE=$?

# Test MAE_VAE model
test_checkpoint "${CHECKPOINT_MAE_VAE}" "RFFR MAE_VAE Model (Fake3)"
MAE_VAE_EXIT_CODE=$?

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
    echo ""
    echo "Each result file contains metrics for all 7 datasets:"
    echo "  - all, stylegan1, stylegan2, stylegan3, styleganxl, sdv1_4, sdv2_1"
else
    echo ""
    echo "✗ Some tests failed!"
    echo "  MAE exit code: ${MAE_EXIT_CODE}"
    echo "  MAE_VAE exit code: ${MAE_VAE_EXIT_CODE}"
fi

exit $(( ${MAE_EXIT_CODE} + ${MAE_VAE_EXIT_CODE} ))
