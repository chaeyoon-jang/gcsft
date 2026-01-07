#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-"/path/to/model"}
EVAL_FILE=${EVAL_FILE:-"/path/to/data.jsonl"}
OUTPUT_DIR=${OUTPUT_DIR:-"logs/seed_samples"}
TEMPERATURE=${TEMPERATURE:-0.7}
TOP_P=${TOP_P:-0.9}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
BATCH_SIZE=${BATCH_SIZE:-0}
USE_CHAT_TEMPLATE=${USE_CHAT_TEMPLATE:-false}
SYSTEM_PROMPT=${SYSTEM_PROMPT:-""}

mkdir -p "${OUTPUT_DIR}"

for seed in $(seq 0 9); do
  output_file="${OUTPUT_DIR}/outputs_seed_${seed}.jsonl"
  python /workspace/gcsft/experiments/eval_vllm.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --eval_file "${EVAL_FILE}" \
    --output_file "${output_file}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
    --batch_size "${BATCH_SIZE}" \
    --seed "${seed}" \
    $( [ "${USE_CHAT_TEMPLATE}" = "true" ] && echo "--use_chat_template" ) \
    $( [ -n "${SYSTEM_PROMPT}" ] && echo "--system_prompt" "${SYSTEM_PROMPT}" )
  echo "Saved ${output_file}"
done
