#!/bin/bash
# Test TRUE RFFR 10F checkpoint (2branch_standard, All Frames, no Fake3)
# Date: Dec 22, 2025

GPU_ID=${1:-"1"}
CHECKPOINT="/andromeda/personal/nmarini/RFFR/rffr_classifier/checkpoint/mae/2branch_standard_F2F_All_2025-10-29-14:10:44_1e297d/best_model/2025-10-29-14:10:44_1e297d/3__AUC_0.99551_270.pth.tar"
FFHQ_ROOT="/oblivion/Datasets/FFHQ"
BATCH_SIZE=128
WORKERS=8
SEED=666
ALL_GENERATORS="all stylegan1-psi-0.5 stylegan2-psi-0.5 stylegan3-psi-0.5 styleganxl-psi-0.5 sdv1_4 sdv2_1"

cd /andromeda/personal/nmarini/RFFR/rffr_classifier || exit 1

echo "================================================================================================"
echo "Testing TRUE RFFR 10F (2branch_standard, All Frames)"
echo "================================================================================================"
echo "Checkpoint: ${CHECKPOINT}"
echo "Start time: $(date)"
echo "================================================================================================"

CUDA_VISIBLE_DEVICES=${GPU_ID} python3 test_ffhq.py \
    --checkpoint "${CHECKPOINT}" \
    --ffhq_root "${FFHQ_ROOT}" \
    --batch_size ${BATCH_SIZE} \
    --workers ${WORKERS} \
    --seed ${SEED} \
    --gpu ${GPU_ID} \
    --generators ${ALL_GENERATORS}

EXIT_CODE=$?

echo ""
echo "================================================================================================"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ TRUE RFFR 10F testing completed successfully!"
    echo "Results saved to: $(dirname ${CHECKPOINT})/test_ffhq_*.txt"
else
    echo "✗ Testing failed with exit code: ${EXIT_CODE}"
fi
echo "End time: $(date)"
echo "================================================================================================"

exit ${EXIT_CODE}
