
set -euo pipefail

PROJECT_ROOT="/data/linux/wkx/IntPhys/references_code/jepa_onn"
PYTHON="/home/linux/anaconda3/envs/semseg/bin/python"
TRAIN="${PROJECT_ROOT}/evaluation_code/evals/intuitive_physics/train_optical.py"
CONFIG="${PROJECT_ROOT}/evaluation_code/evals/intuitive_physics/configs/fsonn_tdm_intphys.yaml"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

for TARGET_NODE in attention_output block_output; do
  "${PYTHON}" "${TRAIN}" \
    --config "${CONFIG}" \
    --experiment-mode realtime_last_node_distillation \
    --target-node "${TARGET_NODE}" \
    --output "/data/linux/wkx/IntPhys/${TARGET_NODE}_distillation.pt" \
    --batch-size 2 \
    --epochs 10 \
    --evaluate-validation-as-test \
    --skip-final-eval
done
