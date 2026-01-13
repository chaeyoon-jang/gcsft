# #!/usr/bin/env bash
# set -euo pipefail

# MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-"meta-llama/Llama-3.2-3B-Instruct"}
OUTPUT_DIR=${OUTPUT_DIR:-"./logs/llama/gsm_seed_samples"}
# INSTRUCTION_TYPE=${INSTRUCTION_TYPE:-"reasoning"}
# TEMPERATURE=${TEMPERATURE:-1.0}
# TOP_P=${TOP_P:-0.9}
# MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-4096}
# TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
# GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
# BATCH_SIZE=${BATCH_SIZE:-32}
USE_CHAT_TEMPLATE=${USE_CHAT_TEMPLATE:-true}
SYSTEM_PROMPT=${SYSTEM_PROMPT:-"You are an expert assistant that provides clear and helpful answers."}

# mkdir -p "${OUTPUT_DIR}"

# for seed in $(seq 0 9); do
#   output_file="${OUTPUT_DIR}/outputs_seed_${seed}.jsonl"
  
#   # Build arguments array to properly handle quotes
#   args=(
#     --model_name_or_path "${MODEL_NAME_OR_PATH}"
#     --instruction_type "${INSTRUCTION_TYPE}"
#     --output_file "${output_file}"
#     --max_new_tokens "${MAX_NEW_TOKENS}"
#     --temperature "${TEMPERATURE}"
#     --top_p "${TOP_P}"
#     --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}"
#     --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}"
#     --batch_size "${BATCH_SIZE}"
#     --seed "${seed}"
#   )
  
#   [ "${USE_CHAT_TEMPLATE}" = "true" ] && args+=(--use_chat_template)
#   [ -n "${SYSTEM_PROMPT}" ] && args+=(--system_prompt "${SYSTEM_PROMPT}")
  
#   python -m experiments.eval_vllm "${args[@]}"
#   echo "Saved ${output_file}"
# done

# Run base model with argmax (greedy decoding, no sampling)
output_file="${OUTPUT_DIR}/outputs_base_argmax.jsonl"

args=(
  --model_name_or_path meta-llama/Llama-3.2-3B-Instruct
  --instruction_type "reasoning"
  --output_file "${output_file}"
  --max_new_tokens 4096
  --temperature 0.0
  --top_p 1.0
  --tensor_parallel_size 1
  --gpu_memory_utilization 0.9
  --batch_size 32
  --seed 0
)

[ "${USE_CHAT_TEMPLATE}" = "true" ] && args+=(--use_chat_template)
[ -n "${SYSTEM_PROMPT}" ] && args+=(--system_prompt "${SYSTEM_PROMPT}")

python -m experiments.eval_vllm "${args[@]}"
echo "Saved ${output_file}"