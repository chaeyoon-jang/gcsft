#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Evaluation script for multiple models and datasets (Test sets)
# ============================================================================
# Models: LLaMA-3.2-3B, Qwen3-8B, GPT-OSS-20B
# Datasets: GSM8K, Ruler 4K, Ruler 8K
# ============================================================================

# Default parameters
# TEMPERATURE=${TEMPERATURE:-0.0}
# TOP_P=${TOP_P:-1.0}
# TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
# GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
# USE_CHAT_TEMPLATE=${USE_CHAT_TEMPLATE:-true}

# # Color codes for output
# GREEN='\033[0;32m'
# BLUE='\033[0;34m'
# YELLOW='\033[1;33m'
# NC='\033[0m' # No Color

# print_header() {
#     echo -e "\n${BLUE}============================================================================${NC}"
#     echo -e "${BLUE}$1${NC}"
#     echo -e "${BLUE}============================================================================${NC}\n"
# }

# run_evaluation() {
#     local model_name=$1
#     local model_path=$2
#     local eval_file=$3
#     local dataset_name=$4
#     local instruction_type=$5
#     local output_dir=$6
#     local max_new_tokens=$7
#     local batch_size=$8
#     local system_prompt=${9:-""}
#     local query_peft_dir=${10:-""}
#     local data_type=${11:-"test"}
#     local output_file="${output_dir}/${model_name}_${dataset_name}_${data_type}.jsonl"
    
#     print_header "Evaluating: ${model_name} on ${dataset_name} (${data_type} Set)"
    
#     mkdir -p "${output_dir}"
    
#     args=(
#         --model_name_or_path "${model_path}"
#         --eval_file "${eval_file}"
#         --instruction_type "${instruction_type}"
#         --output_file "${output_file}"
#         --max_new_tokens "${max_new_tokens}"
#         --batch_size "${batch_size}"
#         --temperature "${TEMPERATURE}"
#         --top_p "${TOP_P}"
#         --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}"
#         --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}"
#         --data_type "${data_type}"
#         --add_conf
#     )
    
#     [ "${USE_CHAT_TEMPLATE}" = "true" ] && args+=(--use_chat_template)
#     [ "${system_prompt}" != "" ] && args+=(--system_prompt "${system_prompt}")
#     [ "${query_peft_dir}" != "" ] && args+=(--query_peft_dir "${query_peft_dir}")
    
#     python -m experiments.eval_vllm "${args[@]}"
    
#     echo -e "${GREEN}✓ Saved: ${output_file}${NC}\n"
# }
# # =============================================================================
# # Extended evaluation for specified checkpoints and datasets
# # =============================================================================

# MODEL_NAME="Llama-3.2-3B-Instruct"
# BASE_MODEL_PATH="meta-llama/Llama-3.2-3B-Instruct"
# OUTPUT_BASE_DIR="./logs/zero_shot_test_evals"

# CHECKPOINTS=(
#     "/workspace/gcsft/logs/Llama-3.2-3B-Instruct_csft_single_gsm_seed0_lr0.0001_kl0.0_bs32*2/checkpoint-600"
#     "/workspace/gcsft/logs/Llama-3.2-3B-Instruct_csft_single_gsm_seed0_lr0.0001_kl0.0_prev/checkpoint-600"
#     "/workspace/gcsft/logs/Llama-3.2-3B-Instruct_csft_single_ruler_4k_seed0_lr0.0001_kl0.0/checkpoint-1000"
#     "/workspace/gcsft/logs/Llama-3.2-3B-Instruct_csft_single_ruler_4k_seed0_lr0.0001_kl1.0/checkpoint-800"
# )

# declare -A DATASETS=(
#     [ruler_4k]="data/processed/ruler_4k_test.jsonl"
#     [ruler_8k]="data/processed/ruler_8k_test.jsonl"
#     [gsm]="openai/gsm8k"
#     [math]="data/processed/math_test.csv"
# )

# declare -A INSTRUCTIONS=(
#     [ruler_4k]="answer_only"
#     [ruler_8k]="answer_only"
#     [gsm]="reasoning"
#     [math]="reasoning"
# )

# declare -A MAX_TOKENS=(
#     [ruler_4k]=4096
#     [ruler_8k]=4096
#     [gsm]=50
#     [math]=50
# )

# declare -A BATCH_SIZES=(
#     [ruler_4k]=32
#     [ruler_8k]=32
#     [gsm]=32
#     [math]=32
# )

# print_header "Starting extended evaluations for ${MODEL_NAME}"

# for ckpt in "${CHECKPOINTS[@]}"; do
#     parent_dir=$(basename "$(dirname "${ckpt}")")
#     ckpt_base=$(basename "${ckpt}")
#     out_dir="${OUTPUT_BASE_DIR}/${parent_dir}_${ckpt_base}"

#     for dataset in ruler_4k ruler_8k gsm math; do
#         eval_file="${DATASETS[$dataset]}"
#         instruction_type="${INSTRUCTIONS[$dataset]}"
#         max_tokens="${MAX_TOKENS[$dataset]}"
#         batch_size="${BATCH_SIZES[$dataset]}"

#         if [ ! -f "${eval_file}" ]; then
#             echo -e "${YELLOW}⚠ Skipping: dataset not found ${eval_file}${NC}"
#             continue
#         fi

#         run_evaluation \
#             "${MODEL_NAME}" \
#             "${BASE_MODEL_PATH}" \
#             "${eval_file}" \
#             "${dataset}" \
#             "${instruction_type}" \
#             "${out_dir}" \
#             "${max_tokens}" \
#             "${batch_size}" \
#             "You are an expert assistant that provides clear and helpful answers." \
#             "${ckpt}"
#     done
# done

# print_header "All evaluations completed"

TEMPERATURE=${TEMPERATURE:-0.0}
TOP_P=${TOP_P:-1.0}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-2}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
USE_CHAT_TEMPLATE=${USE_CHAT_TEMPLATE:-true}

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_header() {
    echo -e "\n${BLUE}============================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================================${NC}\n"
}

run_evaluation() {
    local model_name=$1
    local model_path=$2
    local eval_file=$3
    local dataset_name=$4
    local instruction_type=$5
    local output_dir=$6
    local max_new_tokens=$7
    local batch_size=$8
    local system_prompt=${9:-""}
    local query_peft_dir=${10:-""}
    local data_type=${11:-"test"}
    local output_file="${output_dir}/${model_name}_${dataset_name}_${data_type}.jsonl"
    
    print_header "Evaluating: ${model_name} on ${dataset_name} (${data_type} Set)"
    
    mkdir -p "${output_dir}"
    
    args=(
        --model_name_or_path "${model_path}"
        --eval_file "${eval_file}"
        --instruction_type "${instruction_type}"
        --output_file "${output_file}"
        --max_new_tokens "${max_new_tokens}"
        --batch_size "${batch_size}"
        --temperature "${TEMPERATURE}"
        --top_p "${TOP_P}"
        --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}"
        --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}"
        --data_type "${data_type}"
        --confidence_prompt_name "multi"
        --add_conf
    )
    
    [ "${USE_CHAT_TEMPLATE}" = "true" ] && args+=(--use_chat_template)
    [ "${system_prompt}" != "" ] && args+=(--system_prompt "${system_prompt}")
    [ "${query_peft_dir}" != "" ] && args+=(--query_peft_dir "${query_peft_dir}")
    
    python -m experiments.eval_vllm "${args[@]}"
    
    echo -e "${GREEN}✓ Saved: ${output_file}${NC}\n"
}
# =============================================================================
# Extended evaluation for specified checkpoints and datasets
# =============================================================================

MODEL_NAME="Llama-3.2-3B-Instruct"
BASE_MODEL_PATH="meta-llama/Llama-3.2-3B-Instruct"
OUTPUT_BASE_DIR="./logs/multi_trained_test_evals_final"

CHECKPOINTS=(
    #"/mnt/home/chaeyun-jang/gcsft/logs/llama/ckpt/logs/Llama-3.2-3B-Instruct_csft_multi_seed0_lr0.0001_kl1.0/checkpoint-2000"
    #"/mnt/home/chaeyun-jang/gcsft/logs/llama/ckpt/logs/Llama-3.2-3B-Instruct_csft_multi_seed0_lr0.0001_kl1.0_bs1_gs32_ms2000_ck1/checkpoint-2000"
    #"/mnt/home/chaeyun-jang/gcsft/logs/llama/ckpt/logs/Llama-3.2-3B-Instruct_csft_multi_seed0_lr0.0001_kl1.0_prev/checkpoint-2000"
    #"/mnt/home/chaeyun-jang/gcsft/logs/llama/final/Llama-3.2-3B-Instruct_csft_multi_seed0_lr0.0001_kl1.0_bs1_gs32_ms2000_ck0/checkpoint-2000"
    #"/mnt/home/chaeyun-jang/gcsft/logs/llama/final/Llama-3.2-3B-Instruct_csft_single_gsm_seed0_lr0.0001_kl0.0_bs2_gs16_ms2000_ck0/checkpoint-600"
    #"/mnt/home/chaeyun-jang/gcsft/logs/llama/final/Llama-3.2-3B-Instruct_csft_single_gsm_seed0_lr0.0001_kl0.0_bs2_gs16_ms2000_ck1/checkpoint-800"
    "/mnt/home/chaeyun-jang/gcsft/logs/llama/final/Llama-3.2-3B-Instruct_csft_single_ruler_4k_seed0_lr0.0001_kl0.0_bs2_gs16_ms2000_ck0/checkpoint-800"
)

declare -A DATASETS=(
    [ruler_4k]="data/processed/ruler_4k_test.jsonl"
    [ruler_8k]="data/processed/ruler_8k_test.jsonl"
    [gsm]="openai/gsm8k"
    [math]="data/processed/math_test.csv"
    [contract_nli]="data/processed/contract_nli_test.csv"
)

declare -A INSTRUCTIONS=(
    [ruler_4k]="answer_only"
    [ruler_8k]="answer_only"
    [gsm]="reasoning"
    [math]="reasoning"
    [contract_nli]="reasoning"
)

declare -A MAX_TOKENS=(
    [ruler_4k]=50
    [ruler_8k]=50
    [gsm]=4096
    [math]=4096
    [contract_nli]=4096
)

declare -A BATCH_SIZES=(
    [ruler_4k]=32
    [ruler_8k]=32
    [gsm]=32
    [math]=32
    [contract_nli]=16
)

print_header "Starting extended evaluations for ${MODEL_NAME}"

for ckpt in "${CHECKPOINTS[@]}"; do
    parent_dir=$(basename "$(dirname "${ckpt}")")
    ckpt_base=$(basename "${ckpt}")
    out_dir="${OUTPUT_BASE_DIR}/${parent_dir}_${ckpt_base}"

    for dataset in ruler_4k ruler_8k gsm math contract_nli; do
    #for dataset in contract_nli; do
        eval_file="${DATASETS[$dataset]}"
        instruction_type="${INSTRUCTIONS[$dataset]}"
        max_tokens="${MAX_TOKENS[$dataset]}"
        batch_size="${BATCH_SIZES[$dataset]}"

        #if [ ! -f "${eval_file}" ]; then
        #    echo -e "${YELLOW}⚠ Skipping: dataset not found ${eval_file}${NC}"
        #    continue
        #fi

        run_evaluation \
            "${MODEL_NAME}" \
            "${BASE_MODEL_PATH}" \
            "${eval_file}" \
            "${dataset}" \
            "${instruction_type}" \
            "${out_dir}" \
            "${max_tokens}" \
            "${batch_size}" \
            "You are an expert assistant that provides clear and helpful answers." \
            "${ckpt}"
    done
done

print_header "All evaluations completed"