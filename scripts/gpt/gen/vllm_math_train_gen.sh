# #!/usr/bin/env bash
set -euo pipefail

MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-"openai/gpt-oss-20b"}
EVAL_FILE=${EVAL_FILE:-"data/processed/math_train.csv"}
OUTPUT_DIR=${OUTPUT_DIR:-"./logs/gpt/math_seed_samples"}
INSTRUCTION_TYPE=${INSTRUCTION_TYPE:-"reasoning"}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-0.9}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-4096}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-2}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
BATCH_SIZE=${BATCH_SIZE:-16}
USE_CHAT_TEMPLATE=${USE_CHAT_TEMPLATE:-true}

mkdir -p "${OUTPUT_DIR}"

for seed in $(seq 0 9); do
  output_file="${OUTPUT_DIR}/train_seed_${seed}.jsonl"
  
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
    --seed "${seed}"
  )
  
  [ "${USE_CHAT_TEMPLATE}" = "true" ] && args+=(--use_chat_template)
  
  python -m experiments.eval_vllm "${args[@]}"
  echo "Saved ${output_file}"
done

output_file="${OUTPUT_DIR}/train_base_argmax.jsonl"

args=(
  --model_name_or_path "${MODEL_NAME_OR_PATH}"
  --eval_file "${EVAL_FILE}"
  --instruction_type "${INSTRUCTION_TYPE}"
  --output_file "${output_file}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --temperature 0.0
  --top_p 1.0
  --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}"
  --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}"
  --batch_size "${BATCH_SIZE}"
)

[ "${USE_CHAT_TEMPLATE}" = "true" ] && args+=(--use_chat_template)

python -m experiments.eval_vllm "${args[@]}"
echo "Saved ${output_file}"