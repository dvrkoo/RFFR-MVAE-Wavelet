#!/bin/bash

# Test RFFR Classifier Models on CelebDF and DFD
# Models: Fake100, Fake10, Fake5, Fake3, Fake1

set -e

cd "$(dirname "$0")"

SAMPLES=14000

echo "================================================================================"
echo "Testing RFFR Classifier Models on CelebDF and DFD"
echo "================================================================================"
echo ""

# GPU 0 - CelebDF tests
(
    echo "[GPU 0] Testing on CelebDF dataset"
    echo ""
    
    echo "[GPU 0][CelebDF] Model 1: 100 fake frames"
    python3 test_model.py \
        --checkpoint /seidenas/users/nmarini/classifier_checkpoint/checkpoint/mae/2branch_standard_F2F_All_Fake100_2025-11-10-15:00:06_317d3b/best_model/2025-11-10-15:00:06_317d3b/1__AUC_0.80423_220.pth.tar \
        --datasets CelebDF \
        --samples $SAMPLES \
        --gpu 0 \
        --save-json \
        --output-dir ./test_results
    
    echo "[GPU 0][CelebDF] Model 2: 10 fake frames"
    python3 test_model.py \
        --checkpoint ./checkpoint/mae/2branch_standard_F2F_All_2025-10-29-14:10:44_1e297d/best_model/2025-10-29-14:10:44_1e297d/1__AUC_0.8137_250.pth.tar \
        --datasets CelebDF \
        --samples $SAMPLES \
        --gpu 0 \
        --save-json \
        --output-dir ./test_results
    
    echo "[GPU 0][CelebDF] Model 3: 5 fake frames"
    python3 test_model.py \
        --checkpoint ./checkpoint/mae/2branch_standard_F2F_All_Fake5_2025-10-27-17:53:32_6637c0/1__AUC_0.73855_190.pth.tar \
        --datasets CelebDF \
        --samples $SAMPLES \
        --gpu 0 \
        --save-json \
        --output-dir ./test_results
    
    echo "[GPU 0][CelebDF] Model 4: 3 fake frames"
    python3 test_model.py \
        --checkpoint ./checkpoint/mae/2branch_standard_F2F_All_Fake3_2025-10-27-17:49:08_85bcc4/1__AUC_0.73587_150.pth.tar \
        --datasets CelebDF \
        --samples $SAMPLES \
        --gpu 0 \
        --save-json \
        --output-dir ./test_results
    
    echo "[GPU 0][CelebDF] Model 5: 1 fake frame"
    python3 test_model.py \
        --checkpoint ./checkpoint/mae/2branch_standard_F2F_All_Fake1_2025-10-28-10:16:24_ebaa2f/1__AUC_0.74283_275.pth.tar \
        --datasets CelebDF \
        --samples $SAMPLES \
        --gpu 0 \
        --save-json \
        --output-dir ./test_results
    
    echo "[GPU 0] CelebDF tests complete"
) &

# GPU 1 - DFD tests
(
    echo "[GPU 1] Testing on DFD dataset"
    echo ""
    
    echo "[GPU 1][DFD] Model 1: 100 fake frames"
    python3 test_model.py \
        --checkpoint /seidenas/users/nmarini/classifier_checkpoint/checkpoint/mae/2branch_standard_F2F_All_Fake100_2025-11-10-15:00:06_317d3b/best_model/2025-11-10-15:00:06_317d3b/1__AUC_0.80423_220.pth.tar \
        --datasets DFD \
        --samples $SAMPLES \
        --gpu 1 \
        --save-json \
        --output-dir ./test_results
    
    echo "[GPU 1][DFD] Model 2: 10 fake frames"
    python3 test_model.py \
        --checkpoint ./checkpoint/mae/2branch_standard_F2F_All_2025-10-29-14:10:44_1e297d/best_model/2025-10-29-14:10:44_1e297d/1__AUC_0.8137_250.pth.tar \
        --datasets DFD \
        --samples $SAMPLES \
        --gpu 1 \
        --save-json \
        --output-dir ./test_results
    
    echo "[GPU 1][DFD] Model 3: 5 fake frames"
    python3 test_model.py \
        --checkpoint ./checkpoint/mae/2branch_standard_F2F_All_Fake5_2025-10-27-17:53:32_6637c0/1__AUC_0.73855_190.pth.tar \
        --datasets DFD \
        --samples $SAMPLES \
        --gpu 1 \
        --save-json \
        --output-dir ./test_results
    
    echo "[GPU 1][DFD] Model 4: 3 fake frames"
    python3 test_model.py \
        --checkpoint ./checkpoint/mae/2branch_standard_F2F_All_Fake3_2025-10-27-17:49:08_85bcc4/1__AUC_0.73587_150.pth.tar \
        --datasets DFD \
        --samples $SAMPLES \
        --gpu 1 \
        --save-json \
        --output-dir ./test_results
    
    echo "[GPU 1][DFD] Model 5: 1 fake frame"
    python3 test_model.py \
        --checkpoint ./checkpoint/mae/2branch_standard_F2F_All_Fake1_2025-10-28-10:16:24_ebaa2f/1__AUC_0.74283_275.pth.tar \
        --datasets DFD \
        --samples $SAMPLES \
        --gpu 1 \
        --save-json \
        --output-dir ./test_results
    
    echo "[GPU 1] DFD tests complete"
) &

# Wait for both GPUs to finish
wait

echo ""
echo "================================================================================"
echo "All tests completed!"
echo "================================================================================"
echo ""
echo "Results saved in: ./test_results/"
echo ""
echo "CelebDF Results:"
ls -lh ./test_results/*CelebDF*.json 2>/dev/null | tail -5
echo ""
echo "DFD Results:"
ls -lh ./test_results/*DFD*.json 2>/dev/null | tail -5

