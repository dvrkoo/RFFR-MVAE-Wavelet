#!/bin/bash

# Parallel Testing Script for RFFR Classifier - Wavelet Models
# Runs multiple wavelet model tests simultaneously on 2 GPUs
# Each test saves results to /seidenas/users/nmarini/test_results/ in JSON format

set -e

# Configuration
SAMPLES=14000
OUTPUT_DIR="/seidenas/users/nmarini/test_results/MAE_wavelet"
BATCH_SIZE=16

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Timestamp for this batch of tests
BATCH_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo "Starting parallel testing batch: $BATCH_TIMESTAMP"
echo "Samples per test: $SAMPLES"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Array of checkpoint paths
declare -a CHECKPOINTS=(
    "/andromeda/personal/nmarini/RFFR/rffr_classifier/checkpoint/mae_vae/2branch_standard_F2F_All_Fake1_2025-10-29-14:17:53_67601b/best_model/2025-10-29-14:17:53_67601b/1__AUC_0.71966_230.pth.tar"
    "/andromeda/personal/nmarini/RFFR/rffr_classifier/checkpoint/mae_vae/2branch_standard_F2F_All_Fake3_2025-10-29-11:33:38_c53c5e/best_model/2025-10-29-11:33:38_c53c5e/1__AUC_0.73467_150.pth.tar"
    "/andromeda/personal/nmarini/RFFR/rffr_classifier/checkpoint/mae_vae/2branch_standard_F2F_All_Fake5_2025-10-29-23:29:44_8c4716/best_model/2025-10-29-23:29:44_8c4716/1__AUC_0.74516_270.pth.tar"
    "/andromeda/personal/nmarini/RFFR/rffr_classifier/checkpoint/mae_vae/2branch_standard_F2F_All_2025-10-29-23:29:18_11d3a6/best_model/2025-10-29-23:29:18_11d3a6/1__AUC_0.81188_190.pth.tar"

)

# Array of test descriptions
declare -a DESCRIPTIONS=(
    "Wavelet_F2F_100frames"
    "Wavelet_F2F_10frames"
    "Wavelet_F2F_5frames"
    "Wavelet_F2F_3frames"
    "Wavelet_F2F_1frame"
)

# Function to run a single test
run_test() {
    local checkpoint=$1
    local description=$2
    local gpu=$3
    local test_index=$4
    
    echo "[GPU $gpu] Starting test $test_index: $description"
    echo "[GPU $gpu] Checkpoint: $(basename $checkpoint)"
    
    python3 test_model.py \
        --checkpoint "$checkpoint" \
        --samples $SAMPLES \
        --batch-size $BATCH_SIZE \
        --gpu "$gpu" \
        --save-json \
        --output-dir "$OUTPUT_DIR" \
        2>&1 | tee "$OUTPUT_DIR/log_${BATCH_TIMESTAMP}_${description}_gpu${gpu}.txt"
    
    echo "[GPU $gpu] Completed test $test_index: $description"
    echo ""
}

# Export function so it's available to subshells
export -f run_test
export SAMPLES BATCH_SIZE OUTPUT_DIR BATCH_TIMESTAMP

# Distribute tests across 2 GPUs
# GPU 0: Tests 1-3
# GPU 1: Tests 4-5

echo "==================================================================="
echo "Starting parallel execution"
echo "GPU 0: Tests 1-3 (100, 10, 5 frames)"
echo "GPU 1: Tests 4-5 (3, 1 frame)"
echo "==================================================================="
echo ""

# GPU 0 tests (run in background)
(
    run_test "${CHECKPOINTS[0]}" "${DESCRIPTIONS[0]}" "0" "1"
    run_test "${CHECKPOINTS[1]}" "${DESCRIPTIONS[1]}" "0" "2"
    run_test "${CHECKPOINTS[2]}" "${DESCRIPTIONS[2]}" "0" "3"
) &
GPU0_PID=$!

# GPU 1 tests (run in background)
(
    run_test "${CHECKPOINTS[3]}" "${DESCRIPTIONS[3]}" "1" "4"
    run_test "${CHECKPOINTS[4]}" "${DESCRIPTIONS[4]}" "1" "5"
) &
GPU1_PID=$!

# Wait for both GPU threads to complete
echo "Waiting for GPU 0 tests (PID: $GPU0_PID)..."
echo "Waiting for GPU 1 tests (PID: $GPU1_PID)..."
echo ""

wait $GPU0_PID
GPU0_STATUS=$?
echo "GPU 0 tests completed with status: $GPU0_STATUS"

wait $GPU1_PID
GPU1_STATUS=$?
echo "GPU 1 tests completed with status: $GPU1_STATUS"

echo ""
echo "==================================================================="
echo "All tests completed!"
echo "==================================================================="
echo ""

# Summary of results
echo "Results saved in: $OUTPUT_DIR"
echo ""
echo "JSON results:"
ls -lh "$OUTPUT_DIR"/*.json 2>/dev/null | tail -5 || echo "No JSON files found"
echo ""

echo "Log files:"
ls -lh "$OUTPUT_DIR"/log_${BATCH_TIMESTAMP}*.txt 2>/dev/null || echo "No log files found"
echo ""

# Create summary file
SUMMARY_FILE="$OUTPUT_DIR/batch_summary_${BATCH_TIMESTAMP}.txt"
{
    echo "Batch Test Summary - $BATCH_TIMESTAMP"
    echo "======================================"
    echo ""
    echo "Configuration:"
    echo "  Samples per test: $SAMPLES"
    echo "  Batch size: $BATCH_SIZE"
    echo "  GPUs used: 0, 1"
    echo ""
    echo "Tests completed:"
    for i in "${!DESCRIPTIONS[@]}"; do
        echo "  $((i+1)). ${DESCRIPTIONS[$i]}"
        echo "     Checkpoint: ${CHECKPOINTS[$i]}"
    done
    echo ""
    echo "Exit status:"
    echo "  GPU 0: $GPU0_STATUS"
    echo "  GPU 1: $GPU1_STATUS"
    echo ""
    echo "Output files:"
    ls "$OUTPUT_DIR" | grep "$BATCH_TIMESTAMP" || echo "  None found"
} > "$SUMMARY_FILE"

echo "Batch summary saved to: $SUMMARY_FILE"
echo ""

if [ $GPU0_STATUS -eq 0 ] && [ $GPU1_STATUS -eq 0 ]; then
    echo "✓ All tests completed successfully!"
    exit 0
else
    echo "✗ Some tests failed. Check logs for details."
    exit 1
fi

