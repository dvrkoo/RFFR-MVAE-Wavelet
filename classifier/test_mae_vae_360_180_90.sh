#!/bin/bash

# Test MAE-VAE Models with 360, 180, and 90 frame configurations
# Tests on FaceForensics++ datasets (DF, F2F, FSW, NT, Mixed, FS) + DFD
# This tests the 3-branch wavelet residual MAE-VAE classifier
# Uses 1__ checkpoints (AUC on all forgeries)
# Runs 2 experiments in parallel (GPU 0 and GPU 1)

set -e
cd "$(dirname "$0")"

# Sample configuration
SAMPLES=14000

# Base checkpoint directory
CHECKPOINT_BASE="/seidenas/users/nmarini/classifier_checkpoint/checkpoint/FF_FNmae_vae"

# FF++ datasets to test
FF_DATASETS="DF F2F FSW NT Mixed FS"

echo "================================================================================"
echo "Testing MAE-VAE 3-Branch Wavelet Residual Models - 360/180/90 Frame Configurations"
echo "================================================================================"
echo ""
echo "Config: generative_model_type=mae_vae, use_wavelets=True, wavelet_residual_branch=True"
echo "Frame configurations: 360, 180, 90"
echo "Datasets: FaceForensics++ (DF, F2F, FS, FSW, NT, Mixed) + DFD + CelebDF"
echo "Samples per dataset: $SAMPLES"
echo "Checkpoint type: 1__ (AUC on all forgeries)"
echo "Running 2 experiments in parallel (GPU 0 and GPU 1)"
echo ""

#==============================================================================
# Batch 1: Fake360 (GPU 0) and Fake180 (GPU 1) in parallel - FF++ datasets
#==============================================================================
echo "################################################################################"
echo "# Batch 1/4: Running Fake360 (GPU 0) and Fake180 (GPU 1) on FF++ datasets"
echo "################################################################################"
echo ""

(
    echo "[GPU 0] Starting Model 1/3: MAE-VAE Fake360 (360 fake frames) - FF++"
    python3 test_model.py \
        --checkpoint $CHECKPOINT_BASE/3branch_wavelet_residual_F2F_All_Fake1_360f_2025-12-17-10:19:11_03a75e/best_model/2025-12-17-10:19:11_03a75e/1__AUC_0.76231_30.pth.tar \
        --datasets $FF_DATASETS \
        --samples $SAMPLES \
        --gpu 0 \
        --save-json \
        --output-dir ./test_results
    echo "[GPU 0] Completed Model 1/3: MAE-VAE Fake360 - FF++"
) &

(
    echo "[GPU 1] Starting Model 2/3: MAE-VAE Fake180 (180 fake frames) - FF++"
    python3 test_model.py \
        --checkpoint $CHECKPOINT_BASE/3branch_wavelet_residual_F2F_All_Fake1_180f_2025-12-17-10:20:10_f8bcbb/best_model/2025-12-17-10:20:10_f8bcbb/1__AUC_0.71832_70.pth.tar \
        --datasets $FF_DATASETS \
        --samples $SAMPLES \
        --gpu 1 \
        --save-json \
        --output-dir ./test_results
    echo "[GPU 1] Completed Model 2/3: MAE-VAE Fake180 - FF++"
) &

wait
echo ""
echo "Batch 1/4 completed!"
echo ""

#==============================================================================
# Batch 2: Fake90 (GPU 0) - FF++ datasets
#==============================================================================
echo "################################################################################"
echo "# Batch 2/4: Running Fake90 (GPU 0) on FF++ datasets"
echo "################################################################################"
echo ""

echo "[GPU 0] Starting Model 3/3: MAE-VAE Fake90 (90 fake frames) - FF++"
python3 test_model.py \
    --checkpoint $CHECKPOINT_BASE/3branch_wavelet_residual_F2F_All_Fake1_90f_2025-12-18-08:30:01_985f3c/best_model/2025-12-18-08:30:01_985f3c/1__AUC_0.68969_200.pth.tar \
    --datasets $FF_DATASETS \
    --samples $SAMPLES \
    --gpu 0 \
    --save-json \
    --output-dir ./test_results
echo "[GPU 0] Completed Model 3/3: MAE-VAE Fake90 - FF++"

echo ""
echo "Batch 2/4 completed!"
echo ""

#==============================================================================
# Batch 3: DFD tests - all models in parallel (GPU 0 and GPU 1)
#==============================================================================
echo "################################################################################"
echo "# Batch 3/4: Running DFD tests on GPU 0 and GPU 1"
echo "################################################################################"
echo ""

(
    echo "[GPU 0] Testing MAE-VAE Fake360 on DFD"
    python3 test_model.py \
        --checkpoint $CHECKPOINT_BASE/3branch_wavelet_residual_F2F_All_Fake1_360f_2025-12-17-10:19:11_03a75e/best_model/2025-12-17-10:19:11_03a75e/1__AUC_0.76231_30.pth.tar \
        --datasets DFD \
        --samples $SAMPLES \
        --gpu 0 \
        --save-json \
        --output-dir ./test_results
    
    echo "[GPU 0] Testing MAE-VAE Fake90 on DFD"
    python3 test_model.py \
        --checkpoint $CHECKPOINT_BASE/3branch_wavelet_residual_F2F_All_Fake1_90f_2025-12-18-08:30:01_985f3c/best_model/2025-12-18-08:30:01_985f3c/1__AUC_0.68969_200.pth.tar \
        --datasets DFD \
        --samples $SAMPLES \
        --gpu 0 \
        --save-json \
        --output-dir ./test_results
    
    echo "[GPU 0] Completed DFD tests"
) &

(
    echo "[GPU 1] Testing MAE-VAE Fake180 on DFD"
    python3 test_model.py \
        --checkpoint $CHECKPOINT_BASE/3branch_wavelet_residual_F2F_All_Fake1_180f_2025-12-17-10:20:10_f8bcbb/best_model/2025-12-17-10:20:10_f8bcbb/1__AUC_0.71832_70.pth.tar \
        --datasets DFD \
        --samples $SAMPLES \
        --gpu 1 \
        --save-json \
        --output-dir ./test_results
    
    echo "[GPU 1] Completed DFD tests"
) &

wait
echo ""
echo "Batch 3/4 completed!"
echo ""

#==============================================================================

#==============================================================================
# Batch 4: CelebDF tests - all models in parallel (GPU 0 and GPU 1)
#==============================================================================
echo "################################################################################"
echo "# Batch 4/4: Running CelebDF tests on GPU 0 and GPU 1"
echo "################################################################################"
echo ""

(
    echo "[GPU 0] Testing MAE-VAE Fake360 on CelebDF"
    python3 test_model.py \
        --checkpoint $CHECKPOINT_BASE/3branch_wavelet_residual_F2F_All_Fake1_360f_2025-12-17-10:19:11_03a75e/best_model/2025-12-17-10:19:11_03a75e/1__AUC_0.76231_30.pth.tar \
        --datasets CelebDF \
        --samples $SAMPLES \
        --gpu 0 \
        --save-json \
        --output-dir ./test_results
    
    echo "[GPU 0] Testing MAE-VAE Fake90 on CelebDF"
    python3 test_model.py \
        --checkpoint $CHECKPOINT_BASE/3branch_wavelet_residual_F2F_All_Fake1_90f_2025-12-18-08:30:01_985f3c/best_model/2025-12-18-08:30:01_985f3c/1__AUC_0.68969_200.pth.tar \
        --datasets CelebDF \
        --samples $SAMPLES \
        --gpu 0 \
        --save-json \
        --output-dir ./test_results
    
    echo "[GPU 0] Completed CelebDF tests"
) &

(
    echo "[GPU 1] Testing MAE-VAE Fake180 on CelebDF"
    python3 test_model.py \
        --checkpoint $CHECKPOINT_BASE/3branch_wavelet_residual_F2F_All_Fake1_180f_2025-12-17-10:20:10_f8bcbb/best_model/2025-12-17-10:20:10_f8bcbb/1__AUC_0.71832_70.pth.tar \
        --datasets CelebDF \
        --samples $SAMPLES \
        --gpu 1 \
        --save-json \
        --output-dir ./test_results
    
    echo "[GPU 1] Completed CelebDF tests"
) &

wait
echo ""
echo "Batch 4/4 completed!"
echo ""

# SUMMARY
#==============================================================================
echo "################################################################################"
echo "# ALL MAE-VAE MODELS (360/180/90 frames) TESTED ON ALL DATASETS (including CelebDF)!"
echo "################################################################################"
echo ""
echo "Results saved to ./test_results/"
echo ""
echo "Recent results:"
ls -lth ./test_results/*.json 2>/dev/null | head -20
echo ""
