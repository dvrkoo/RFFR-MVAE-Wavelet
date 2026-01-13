#!/bin/bash

# Test MAE-VAE 3-branch Wavelet Models on DFD dataset
# Models: Fake10, Fake5, Fake3, Fake1

set -e
cd "$(dirname "$0")"

SAMPLES=14000

echo "================================================================================"
echo "Testing MAE-VAE 3-Branch Wavelet Models on DFD"
echo "================================================================================"
echo ""
echo "Config: generative_model_type=mae_vae, use_wavelets=True, separate_wavelet_branch=True"
echo ""

# Test all 4 models sequentially on DFD
echo "Model 1/4: Fake10 (10 fake frames) - 3-branch wavelet MAE-VAE"
python3 test_model.py \
    --checkpoint ./checkpoint/mae_vae/3branch_wavelet_residual_F2F_All_2025-10-31-15:30:50_85d8e3/best_model/2025-10-31-15:30:50_85d8e3/3__AUC_0.99622_150.pth.tar \
    --datasets DFD \
    --samples $SAMPLES \
    --gpu 1 \
    --save-json \
    --output-dir ./test_results

echo ""
echo "Model 2/4: Fake5 (5 fake frames) - 3-branch wavelet MAE-VAE"
python3 test_model.py \
    --checkpoint ./checkpoint/mae_vae/3branch_wavelet_residual_F2F_All_Fake5_2025-10-31-09:14:15_d0252e/best_model/2025-10-31-09:14:15_d0252e/3__AUC_0.9974_290.pth.tar \
    --datasets DFD \
    --samples $SAMPLES \
    --gpu 1 \
    --save-json \
    --output-dir ./test_results

echo ""
echo "Model 3/4: Fake3 (3 fake frames) - 3-branch wavelet MAE-VAE"
python3 test_model.py \
    --checkpoint ./checkpoint/mae_vae/3branch_wavelet_residual_F2F_All_Fake3_2025-10-31-09:08:12_2004fc/best_model/2025-10-31-09:08:12_2004fc/3__AUC_0.9948_270.pth.tar \
    --datasets DFD \
    --samples $SAMPLES \
    --gpu 1 \
    --save-json \
    --output-dir ./test_results

echo ""
echo "Model 4/4: Fake1 (1 fake frame) - 3-branch wavelet MAE-VAE"
python3 test_model.py \
    --checkpoint ./checkpoint/mae_vae/3branch_wavelet_residual_F2F_All_Fake1_2025-10-30-14:16:57_11bd41/best_model/2025-10-30-14:16:57_11bd41/3__AUC_0.98903_170.pth.tar \
    --datasets DFD \
    --samples $SAMPLES \
    --gpu 1 \
    --save-json \
    --output-dir ./test_results

echo ""
echo "================================================================================"
echo "All 4 MAE-VAE wavelet models tested on DFD!"
echo "================================================================================"
echo ""
echo "Results:"
ls -lth ./test_results/*DFD*.json 2>/dev/null | head -10

