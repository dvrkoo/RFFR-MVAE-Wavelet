#!/bin/bash

# Test MAE Models with all frame configurations on FaceForensics++ datasets
# Configurations: Fake1, Fake3, Fake5, All (default), Fake100
# This tests the 2-branch standard MAE classifier
# Uses 1__ checkpoints (AUC on all forgeries)
# Runs 2 experiments in parallel (GPU 0 and GPU 1)

set -e
cd "$(dirname "$0")"

# Sample configuration
SAMPLES=14000

# Base checkpoint directory
CHECKPOINT_BASE="/seidenas/users/nmarini/classifier_checkpoint/checkpoint/mae"

# FF++ datasets to test
DATASETS="FSW"

echo "================================================================================"
echo "Testing MAE 2-Branch Standard Models - All Frame Configurations"
echo "================================================================================"
echo ""
echo "Config: generative_model_type=mae, 2-branch standard"
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
    echo "[GPU 0] Starting Model 1/5: MAE Fake1 (1 fake frame)"
    python3 test_model.py \
        --checkpoint $CHECKPOINT_BASE/2branch_standard_DF_All_Fake1_2025-12-12-10:10:55_fd4c21/best_model/2025-12-12-10:10:55_fd4c21/1__AUC_0.73417_90.pth.tar \
        --datasets $DATASETS \
        --samples $SAMPLES \
        --gpu 0 \
        --save-json \
        --output-dir ./test_results
    echo "[GPU 0] Completed Model 1/5: MAE Fake1"
) &

(
    echo "[GPU 1] Starting Model 2/5: MAE Fake3 (3 fake frames)"
    python3 test_model.py \
        --checkpoint $CHECKPOINT_BASE/2branch_standard_DF_All_Fake3_2025-12-12-10:10:38_4f1af7/best_model/2025-12-12-10:10:38_4f1af7/1__AUC_0.75774_220.pth.tar \
        --datasets $DATASETS \
        --samples $SAMPLES \
        --gpu 1 \
        --save-json \
        --output-dir ./test_results
    echo "[GPU 1] Completed Model 2/5: MAE Fake3"
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
    echo "[GPU 0] Starting Model 3/5: MAE Fake5 (5 fake frames)"
    python3 test_model.py \
        --checkpoint $CHECKPOINT_BASE/2branch_standard_DF_All_Fake5_2025-12-11-18:00:15_9e901a/best_model/2025-12-11-18:00:15_9e901a/1__AUC_0.72774_90.pth.tar \
        --datasets $DATASETS \
        --samples $SAMPLES \
        --gpu 0 \
        --save-json \
        --output-dir ./test_results
    echo "[GPU 0] Completed Model 3/5: MAE Fake5"
) &

(
    echo "[GPU 1] Starting Model 4/5: MAE All (all frames)"
    python3 test_model.py \
        --checkpoint $CHECKPOINT_BASE/2branch_standard_DF_All_2025-12-12-15:56:34_6c93e6/best_model/2025-12-12-15:56:34_6c93e6/1__AUC_0.76009_210.pth.tar \
        --datasets $DATASETS \
        --samples $SAMPLES \
        --gpu 1 \
        --save-json \
        --output-dir ./test_results
    echo "[GPU 1] Completed Model 4/5: MAE All"
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

echo "[GPU 0] Starting Model 5/5: MAE Fake100 (100 fake frames)"
python3 test_model.py \
    --checkpoint $CHECKPOINT_BASE/2branch_standard_DF_All_Fake100_2025-12-14-12:17:35_a29a1a/best_model/2025-12-14-12:17:35_a29a1a/1__AUC_0.74027_185.pth.tar \
    --datasets $DATASETS \
    --samples $SAMPLES \
    --gpu 0 \
    --save-json \
    --output-dir ./test_results
echo "[GPU 0] Completed Model 5/5: MAE Fake100"

echo ""
echo "Batch 3/3 completed!"
echo ""

#==============================================================================
# SUMMARY
#==============================================================================
echo "################################################################################"
echo "# ALL MAE MODELS TESTED ON FACEFORENSICS++ DATASETS!"
echo "################################################################################"
echo ""
echo "Results saved to ./test_results/"
echo ""
echo "Recent results:"
ls -lth ./test_results/*.json 2>/dev/null | head -15
echo ""
