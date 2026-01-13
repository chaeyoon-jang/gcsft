#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-"Qwen/Qwen3-8B"}
EVAL_FILE=${EVAL_FILE:-"data/processed/math_test.csv"}
OUTPUT_DIR=${OUTPUT_DIR:-"./logs/qwen/math_4k_test_samples"}
OUTPUT_FILE=${OUTPUT_FILE:-}
INSTRUCTION_TYPE=${INSTRUCTION_TYPE:-"reasoning"}
TEMPERATURE=${TEMPERATURE:-0.0}
TOP_P=${TOP_P:-1.0}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-4096}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
BATCH_SIZE=${BATCH_SIZE:-2}
USE_CHAT_TEMPLATE=${USE_CHAT_TEMPLATE:-true}

mkdir -p "${OUTPUT_DIR}"

# Determine output file (honors explicit OUTPUT_FILE)
if [[ -n "${OUTPUT_FILE}" ]]; then
  output_file="${OUTPUT_FILE}"
else
  output_file="${OUTPUT_DIR}/outputs.jsonl"
fi

# Build arguments array to properly handle quotes
args=(
  --model_name_or_path "${MODEL_NAME_OR_PATH}"
  --eval_file "${EVAL_FILE}"
  --instruction_type "${INSTRUCTION_TYPE}"
  --output_file "${output_file}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --temperature "${TEMPERATURE}"
  --top_p "${TOP_P}"
  --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}"
  --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}"
  --batch_size "${BATCH_SIZE}"
)

[ "${USE_CHAT_TEMPLATE}" = "true" ] && args+=(--use_chat_template)

python -m experiments.eval_vllm "${args[@]}"
echo "Saved ${output_file}"