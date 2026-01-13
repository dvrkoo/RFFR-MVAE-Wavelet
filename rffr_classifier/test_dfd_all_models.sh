#!/bin/bash

# Test all 5 RFFR models on DFD dataset

set -e
cd "$(dirname "$0")"

SAMPLES=14000

echo "================================================================================"
echo "Testing RFFR Classifier Models on DFDC (5 models)"
echo "================================================================================"
echo ""

# Test all 5 models sequentially on DFD
echo "Model 1/5: Fake100 (100 fake frames per video)"
python3 test_model.py \
    --checkpoint /seidenas/users/nmarini/classifier_checkpoint/checkpoint/mae/2branch_standard_F2F_All_Fake100_2025-11-10-15:00:06_317d3b/best_model/2025-11-10-15:00:06_317d3b/1__AUC_0.80423_220.pth.tar \
    --datasets DFDC \
    --samples $SAMPLES \
    --gpu 0 \
    --save-json \
    --output-dir ./test_results

echo ""
echo "Model 2/5: Fake10 (10 fake frames per video)"
python3 test_model.py \
    --checkpoint ./checkpoint/mae/2branch_standard_F2F_All_2025-10-29-14:10:44_1e297d/best_model/2025-10-29-14:10:44_1e297d/1__AUC_0.8137_250.pth.tar \
    --datasets DFDC \
    --samples $SAMPLES \
    --gpu 0 \
    --save-json \
    --output-dir ./test_results

echo ""
echo "Model 3/5: Fake5 (5 fake frames per video)"
python3 test_model.py \
    --checkpoint ./checkpoint/mae/2branch_standard_F2F_All_Fake5_2025-10-27-17:53:32_6637c0/1__AUC_0.73855_190.pth.tar \
    --datasets DFDC \
    --samples $SAMPLES \
    --gpu 0 \
    --save-json \
    --output-dir ./test_results

echo ""
echo "Model 4/5: Fake3 (3 fake frames per video)"
python3 test_model.py \
    --checkpoint ./checkpoint/mae/2branch_standard_F2F_All_Fake3_2025-10-27-17:49:08_85bcc4/1__AUC_0.73587_150.pth.tar \
    --datasets DFDC \
    --samples $SAMPLES \
    --gpu 0 \
    --save-json \
    --output-dir ./test_results

echo ""
echo "Model 5/5: Fake1 (1 fake frame per video)"
python3 test_model.py \
    --checkpoint ./checkpoint/mae/2branch_standard_F2F_All_Fake1_2025-10-28-10:16:24_ebaa2f/1__AUC_0.74283_275.pth.tar \
    --datasets DFDC \
    --samples $SAMPLES \
    --gpu 0 \
    --save-json \
    --output-dir ./test_results

echo ""
echo "================================================================================"
echo "All 5 models tested on DFDC!"
echo "================================================================================"
echo ""
echo "Results:"
ls -lh ./test_results/*DFD*.json 2>/dev/null | tail -5

