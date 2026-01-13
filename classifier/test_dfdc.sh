#!/bin/bash

# Test RFFR model on DFDC dataset

set -e
cd "$(dirname "$0")"

# Default values
SAMPLES=13000
GPU="0"
CHECKPOINT=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --samples)
            SAMPLES="$2"
            shift 2
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --help)
            echo "Usage: ./test_dfdc.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --checkpoint PATH    Path to model checkpoint (default: use latest)"
            echo "  --samples NUM        Number of samples to test (default: 13000)"
            echo "  --gpu ID             GPU device ID (default: 0)"
            echo "  --help               Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./test_dfdc.sh"
            echo "  ./test_dfdc.sh --samples 7000"
            echo "  ./test_dfdc.sh --checkpoint ./checkpoint/mae/run_name/best_model.pth.tar"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "================================================================================"
echo "Testing RFFR Classifier on DFDC Dataset"
echo "================================================================================"
echo ""
echo "Configuration:"
echo "  Samples: $SAMPLES"
echo "  GPU: $GPU"
if [ -n "$CHECKPOINT" ]; then
    echo "  Checkpoint: $CHECKPOINT"
else
    echo "  Checkpoint: Latest model (auto-detected)"
fi
echo ""

# Build command
CMD="python3 test_model.py --datasets DFDC --samples $SAMPLES --gpu $GPU --save-json --output-dir ./test_results"

if [ -n "$CHECKPOINT" ]; then
    CMD="$CMD --checkpoint $CHECKPOINT"
fi

echo "Running: $CMD"
echo ""

# Run the test
eval $CMD

echo ""
echo "================================================================================"
echo "DFDC testing completed!"
echo "================================================================================"
echo ""
echo "Results saved to: ./test_results/"
ls -lh ./test_results/*DFDC*.json 2>/dev/null | tail -1 || echo "No JSON results found"
