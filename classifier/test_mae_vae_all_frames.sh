#!/bin/bash

# Test MAE-VAE Models with all frame configurations on FaceForensics++ datasets
# Configurations: Fake1, Fake3, Fake5, All (default), Fake100
# This tests the 3-branch wavelet residual MAE-VAE classifier
# Uses 1__ checkpoints (AUC on all forgeries)
# Runs 2 experiments in parallel (GPU 0 and GPU 1)

set -e
cd "$(dirname "$0")"

# Sample configuration
SAMPLES=14000

# Base checkpoint directory
CHECKPOINT_BASE="/seidenas/users/nmarini/classifier_checkpoint/checkpoint/mae_vae"

# FF++ datasets to test
DATASETS="DF F2F FSW NT Mixed FS"

echo "================================================================================"
echo "Testing MAE-VAE 3-Branch Wavelet Residual Models - All Frame Configurations"
echo "================================================================================"
echo ""
echo "Config: generative_model_type=mae_vae, use_wavelets=True, separate_wavelet_branch=True"
echo "Frame configurations: Fake1, Fake3, Fake5, All, Fake100"
echo "Datasets: FaceForensics++ (DF, F2F, FS, NT, Mixed)"
echo "Samples per dataset: $SAMPLES"
echo "Checkpoint type: 1__ (AUC on all forgeries)"
echo "Running 2 experiments in parallel (GPU 0 and GPU 1)"
echo ""

#==============================================================================
# Batch 1: Fake1 (GPU 0) and Fake3 (GPU 1) in parallel
#==============================================================================
echo "################################################################################"
echo "# Batch 1/3: Running Fake1 (GPU 0) and Fake3 (GPU 1) in parallel"
echo "################################################################################"
echo ""

(
    echo "[GPU 0] Starting Model 1/5: MAE-VAE Fake1 (1 fake frame)"
    python3 test_model.py \
        --checkpoint $CHECKPOINT_BASE/3branch_wavelet_residual_DF_All_Fake1_2025-12-13-10:54:25_5787e5/best_model/2025-12-13-10:54:25_5787e5/1__AUC_0.72487_140.pth.tar \
        --datasets $DATASETS \
        --samples $SAMPLES \
        --gpu 0 \
        --save-json \
        --output-dir ./test_results
    echo "[GPU 0] Completed Model 1/5: MAE-VAE Fake1"
) &

(
    echo "[GPU 1] Starting Model 2/5: MAE-VAE Fake3 (3 fake frames)"
    python3 test_model.py \
        --checkpoint $CHECKPOINT_BASE/3branch_wavelet_residual_DF_All_Fake3_2025-12-13-09:58:52_cc20fb/best_model/2025-12-13-09:58:52_cc20fb/1__AUC_0.74566_120.pth.tar \
        --datasets $DATASETS \
        --samples $SAMPLES \
        --gpu 1 \
        --save-json \
        --output-dir ./test_results
    echo "[GPU 1] Completed Model 2/5: MAE-VAE Fake3"
) &

wait
echo ""
echo "Batch 1/3 completed!"
echo ""

#==============================================================================
# Batch 2: Fake5 (GPU 0) and All (GPU 1) in parallel
#==============================================================================
echo "################################################################################"
echo "# Batch 2/3: Running Fake5 (GPU 0) and All (GPU 1) in parallel"
echo "################################################################################"
echo ""

(
    echo "[GPU 0] Starting Model 3/5: MAE-VAE Fake5 (5 fake frames)"
    python3 test_model.py \
        --checkpoint $CHECKPOINT_BASE/3branch_wavelet_residual_DF_All_Fake5_2025-12-13-09:58:33_98c107/best_model/2025-12-13-09:58:33_98c107/1__AUC_0.72745_100.pth.tar \
        --datasets $DATASETS \
        --samples $SAMPLES \
        --gpu 0 \
        --save-json \
        --output-dir ./test_results
    echo "[GPU 0] Completed Model 3/5: MAE-VAE Fake5"
) &

(
    echo "[GPU 1] Starting Model 4/5: MAE-VAE All (all frames)"
    python3 test_model.py \
        --checkpoint $CHECKPOINT_BASE/3branch_wavelet_residual_DF_All_2025-12-12-17:56:23_6fcd51/best_model/2025-12-12-17:56:23_6fcd51/1__AUC_0.71958_80.pth.tar \
        --datasets $DATASETS \
        --samples $SAMPLES \
        --gpu 1 \
        --save-json \
        --output-dir ./test_results
    echo "[GPU 1] Completed Model 4/5: MAE-VAE All"
) &

wait
echo ""
echo "Batch 2/3 completed!"
echo ""

#==============================================================================
# Batch 3: Fake100 (GPU 0) - single experiment
#==============================================================================
echo "################################################################################"
echo "# Batch 3/3: Running Fake100 (GPU 0)"
echo "################################################################################"
echo ""

echo "[GPU 0] Starting Model 5/5: MAE-VAE Fake100 (100 fake frames)"
python3 test_model.py \
    --checkpoint $CHECKPOINT_BASE/3branch_wavelet_residual_DF_All_Fake100_2025-12-14-12:16:45_ef95f8/best_model/2025-12-14-12:16:45_ef95f8/1__AUC_0.74949_110.pth.tar \
    --datasets $DATASETS \
    --samples $SAMPLES \
    --gpu 0 \
    --save-json \
    --output-dir ./test_results
echo "[GPU 0] Completed Model 5/5: MAE-VAE Fake100"

echo ""
echo "Batch 3/3 completed!"
echo ""

#==============================================================================
# SUMMARY
#==============================================================================
echo "################################################################################"
echo "# ALL MAE-VAE MODELS TESTED ON FACEFORENSICS++ DATASETS!"
echo "################################################################################"
echo ""
echo "Results saved to ./test_results/"
echo ""
echo "Recent results:"
ls -lth ./test_results/*.json 2>/dev/null | head -15
echo ""
