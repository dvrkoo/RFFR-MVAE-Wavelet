#!/bin/bash

# Simple Parallel Testing - Modified from test_models.sh
# Runs tests on 2 GPUs with 14000 samples each
# Automatically saves JSON results to ./test_results/

set -e

cd "$(dirname "$0")"

SAMPLES=14000

echo "Starting parallel testing with $SAMPLES samples per test"
echo ""

# GPU 0 - Tests 1, 2, 3
(
    echo "[GPU 0] Starting test 1: 100 frames"
    python3 test_model.py --checkpoint /seidenas/users/nmarini/classifier_checkpoint/checkpoint/mae/2branch_standard_F2F_All_Fake100_2025-11-10-15:00:06_317d3b/best_model/2025-11-10-15:00:06_317d3b/1__AUC_0.80423_220.pth.tar --samples $SAMPLES --gpu 0 --save-json --output-dir ../test_results
    
    echo "[GPU 0] Starting test 2: 10 frames"
    python3 test_model.py --checkpoint ./checkpoint/mae/2branch_standard_F2F_All_2025-10-29-14:10:44_1e297d/best_model/2025-10-29-14:10:44_1e297d/1__AUC_0.8137_250.pth.tar --samples $SAMPLES --gpu 0 --save-json --output-dir ../test_results
    
    echo "[GPU 0] Starting test 3: 5 frames"
    python3 test_model.py --checkpoint ./checkpoint/mae/2branch_standard_F2F_All_Fake5_2025-10-27-17:53:32_6637c0/1__AUC_0.73855_190.pth.tar --samples $SAMPLES --gpu 0 --save-json --output-dir ../test_results
    
    echo "[GPU 0] All tests complete"
) &

# GPU 1 - Tests 4, 5
(
    echo "[GPU 1] Starting test 4: 3 frames"
    python3 test_model.py --checkpoint ./checkpoint/mae/2branch_standard_F2F_All_Fake3_2025-10-27-17:49:08_85bcc4/1__AUC_0.73587_150.pth.tar --samples $SAMPLES --gpu 1 --save-json --output-dir ../test_results
    
    echo "[GPU 1] Starting test 5: 1 frame"
    python3 test_model.py --checkpoint ./checkpoint/mae/2branch_standard_F2F_All_Fake1_2025-10-28-10:16:24_ebaa2f/1__AUC_0.74283_275.pth.tar --samples $SAMPLES --gpu 1 --save-json --output-dir ../test_results
    
    echo "[GPU 1] All tests complete"
) &

# Wait for both to finish
wait

echo ""
echo "All parallel tests completed!"
echo "Results saved in: ../test_results/"
ls -lh ../test_results/*.json 2>/dev/null | tail -5

