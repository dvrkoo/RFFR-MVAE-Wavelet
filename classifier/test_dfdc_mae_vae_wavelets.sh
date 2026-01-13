#!/bin/bash
# Test MAE-VAE 3-branch Wavelet Models on DFDC dataset
# Same 4 checkpoints as DFD script, evaluated on DFDC
set -e
cd "$(dirname "$0")"
# DFDC has only 860 real samples, so we balance with 860 fake samples
SAMPLES=860
echo "================================================================================"
echo "Testing MAE-VAE 3-Branch Wavelet Models on DFDC"
echo "================================================================================"
echo ""
echo "Config: generative_model_type=mae_vae, use_wavelets=True, separate_wavelet_branch=True"
echo ""
# Model 1/4: Fake10
echo "Model 1/4: Fake10 (10 fake frames) - 3-branch wavelet MAE-VAE on DFDC"
python3 test_model.py \
    --checkpoint ./checkpoint/mae_vae/3branch_wavelet_residual_F2F_All_2025-10-31-15:30:50_85d8e3/best_model/2025-10-31-15:30:50_85d8e3/3__AUC_0.99622_150.pth.tar \
    --datasets DFDC \
    --samples $SAMPLES \
    --gpu 1 \
    --save-json \
    --output-dir ./test_results
echo ""
# Model 2/4: Fake5
echo "Model 2/4: Fake5 (5 fake frames) - 3-branch wavelet MAE-VAE on DFDC"
python3 test_model.py \
    --checkpoint ./checkpoint/mae_vae/3branch_wavelet_residual_F2F_All_Fake5_2025-10-31-09:14:15_d0252e/best_model/2025-10-31-09:14:15_d0252e/3__AUC_0.9974_290.pth.tar \
    --datasets DFDC \
    --samples $SAMPLES \
    --gpu 1 \
    --save-json \
    --output-dir ./test_results
echo ""
# Model 3/4: Fake3
echo "Model 3/4: Fake3 (3 fake frames) - 3-branch wavelet MAE-VAE on DFDC"
python3 test_model.py \
    --checkpoint ./checkpoint/mae_vae/3branch_wavelet_residual_F2F_All_Fake3_2025-10-31-09:08:12_2004fc/best_model/2025-10-31-09:08:12_2004fc/3__AUC_0.9948_270.pth.tar \
    --datasets DFDC \
    --samples $SAMPLES \
    --gpu 1 \
    --save-json \
    --output-dir ./test_results
echo ""
# Model 4/4: Fake1
echo "Model 4/4: Fake1 (1 fake frame) - 3-branch wavelet MAE-VAE on DFDC"
python3 test_model.py \
    --checkpoint ./checkpoint/mae_vae/3branch_wavelet_residual_F2F_All_Fake1_2025-10-30-14:16:57_11bd41/best_model/2025-10-30-14:16:57_11bd41/3__AUC_0.98903_170.pth.tar \
    --datasets DFDC \
    --samples $SAMPLES \
    --gpu 1 \
    --save-json \
    --output-dir ./test_results
echo ""
echo "================================================================================"
echo "All 4 MAE-VAE wavelet models tested on DFDC!"
echo "================================================================================"
echo ""
echo "Results:"
ls -lth ./test_results/*DFDC*.json 2>/dev/null | head -10
