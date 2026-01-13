#!/bin/bash

# Test MAE Models with 360, 180, and 90 frame configurations
# Tests on DFD and CelebDF (FF++ results already calculated)
# This tests the 2-branch standard MAE classifier
# Uses 1__ checkpoints (AUC on all forgeries)
# Runs tests in parallel on GPU 0 and GPU 1

set -e
cd "$(dirname "$0")"

# Sample configuration
SAMPLES=14000

# Base checkpoint directory
CHECKPOINT_BASE="/seidenas/users/nmarini/classifier_checkpoint/checkpoint/FF_FNmae"

echo "================================================================================"
echo "Testing MAE 2-Branch Standard Models - 360/180/90 Frame Configurations"
echo "================================================================================"
echo ""
echo "Config: generative_model_type=mae, 2-branch standard"
echo "Frame configurations: 360, 180, 90"
echo "Datasets: DFD + CelebDF (FF++ results already calculated)"
echo "Samples per dataset: $SAMPLES"
echo "Checkpoint type: 1__ (AUC on all forgeries)"
echo "Running tests in parallel on GPU 0 and GPU 1"
echo ""

#==============================================================================
# DFD tests - all 3 models in parallel (GPU 0 and GPU 1)
#==============================================================================
echo "################################################################################"
echo "# Batch 1/2: Running DFD tests on GPU 0 and GPU 1"
echo "################################################################################"
echo ""
#
# (
#     echo "[GPU 0] Testing MAE Fake360 on DFD"
#     python3 test_model.py \
#         --checkpoint $CHECKPOINT_BASE/2branch_standard_F2F_All_Fake1_360f_2025-12-15-12:31:43_84a3ef/best_model/2025-12-15-12:31:43_84a3ef/1__AUC_0.7554_100.pth.tar \
#         --datasets DFD \
#         --samples $SAMPLES \
#         --gpu 1 \
#         --save-json \
#         --output-dir ./test_results
#
#     echo "[GPU 0] Testing MAE Fake90 on DFD"
#     python3 test_model.py \
#         --checkpoint $CHECKPOINT_BASE/2branch_standard_F2F_All_Fake1_90f_2025-12-17-09:32:39_d09531/best_model/2025-12-17-09:32:39_d09531/1__AUC_0.72584_235.pth.tar \
#         --datasets DFD \
#         --samples $SAMPLES \
#         --gpu 1 \
#         --save-json \
#         --output-dir ./test_results
#
#     echo "[GPU 0] Completed DFD tests"
# ) &
#
# (
#     echo "[GPU 1] Testing MAE Fake180 on DFD"
#     python3 test_model.py \
#         --checkpoint $CHECKPOINT_BASE/2branch_standard_F2F_All_Fake1_180f_2025-12-15-17:12:16_6c405e/best_model/2025-12-15-17:12:16_6c405e/1__AUC_0.7254_60.pth.tar \
#         --datasets DFD \
#         --samples $SAMPLES \
#         --gpu 1 \
#         --save-json \
#         --output-dir ./test_results
#
#     echo "[GPU 1] Completed DFD tests"
# ) &
#
# wait
# echo ""
# echo "Batch 1/2 completed!"
# echo ""

#==============================================================================
# CelebDF tests - all 3 models in parallel (GPU 0 and GPU 1)
#==============================================================================
echo "################################################################################"
echo "# Batch 2/2: Running CelebDF tests on GPU 0 and GPU 1"
echo "################################################################################"
echo ""

(
    echo "[GPU 0] Testing MAE Fake360 on CelebDF"
    python3 test_model.py \
        --checkpoint $CHECKPOINT_BASE/2branch_standard_F2F_All_Fake1_360f_2025-12-15-12:31:43_84a3ef/best_model/2025-12-15-12:31:43_84a3ef/1__AUC_0.7554_100.pth.tar \
        --datasets CelebDF \
        --samples $SAMPLES \
        --gpu 1 \
        --save-json \
        --output-dir ./test_results
    
    echo "[GPU 0] Testing MAE Fake90 on CelebDF"
    python3 test_model.py \
        --checkpoint $CHECKPOINT_BASE/2branch_standard_F2F_All_Fake1_90f_2025-12-17-09:32:39_d09531/best_model/2025-12-17-09:32:39_d09531/1__AUC_0.72584_235.pth.tar \
        --datasets CelebDF \
        --samples $SAMPLES \
        --gpu 1 \
        --save-json \
        --output-dir ./test_results
    
    echo "[GPU 0] Completed CelebDF tests"
) &

(
    echo "[GPU 1] Testing MAE Fake180 on CelebDF"
    python3 test_model.py \
        --checkpoint $CHECKPOINT_BASE/2branch_standard_F2F_All_Fake1_180f_2025-12-15-17:12:16_6c405e/best_model/2025-12-15-17:12:16_6c405e/1__AUC_0.7254_60.pth.tar \
        --datasets CelebDF \
        --samples $SAMPLES \
        --gpu 1 \
        --save-json \
        --output-dir ./test_results
    
    echo "[GPU 1] Completed CelebDF tests"
) &

wait
echo ""
echo "Batch 2/2 completed!"
echo ""

#==============================================================================
# SUMMARY
#==============================================================================
echo "################################################################################"
echo "# ALL MAE MODELS (360/180/90 frames) TESTED ON DFD AND CelebDF!"
echo "################################################################################"
echo ""
echo "Results saved to ./test_results/"
echo ""
echo "Recent DFD Results:"
ls -lth ./test_results/*DFD*.json 2>/dev/null | head -10
echo ""
echo "Recent CelebDF Results:"
ls -lth ./test_results/*CelebDF*.json 2>/dev/null | head -10
echo ""
