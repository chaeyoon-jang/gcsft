#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Evaluation script for multiple models and datasets (Test sets)
# ============================================================================
# Models: LLaMA-3.2-3B, Qwen3-8B, GPT-OSS-20B
# Datasets: GSM8K, Ruler 4K, Ruler 8K
# ============================================================================

# Default parameters
TEMPERATURE=${TEMPERATURE:-0.0}
TOP_P=${TOP_P:-1.0}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
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
    
    local output_file="${output_dir}/${model_name}_${dataset_name}_test.jsonl"
    
    print_header "Evaluating: ${model_name} on ${dataset_name} (Test Set)"
    
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
        --add_conf
        --data_type test
    )
    
    [ "${USE_CHAT_TEMPLATE}" = "true" ] && args+=(--use_chat_template)
    [ "${system_prompt}" != "" ] && args+=(--system_prompt "${system_prompt}")
    
    python -m experiments.eval_vllm "${args[@]}"
    
    echo -e "${GREEN}✓ Saved: ${output_file}${NC}\n"
}

# ============================================================================
# 1. LLaMA-3.2-3B Evaluations
# ============================================================================
# print_header "LLaMA-3.2-3B Evaluations"

run_evaluation \
    "llama_3b" \
    "meta-llama/Llama-3.2-3B-Instruct" \
    "data/processed/mmlu_test.csv" \
    "mmlu" \
    "answer_only" \
    "logs/zero_shot_test_evals" \
    50 \
    16 \
    "You are an expert assistant that provides clear and helpful answers."

# run_evaluation \
#     "llama_3b" \
#     "meta-llama/Llama-3.2-3B-Instruct" \
#     "openai/gsm8k" \
#     "gsm" \
#     "reasoning" \
#     "logs/zero_shot_test_evals" \
#     4096 \
#     32 \
#     "You are an expert assistant that provides clear and helpful answers."

# ============================================================================
# 2. Qwen3-8B Evaluations
# ============================================================================
# print_header "Qwen3-8B Evaluations"

# run_evaluation \
#     "qwen_8b" \
#     "Qwen/Qwen3-8B" \
#     "data/processed/math_test.csv" \
#     "math" \
#     "reasoning" \
#     "logs/zero_shot_test_evals" \
#     4096 \
#     16

# run_evaluation \
#     "qwen_8b" \
#     "Qwen/Qwen3-8B" \
#     "data/processed/ruler_8k_test.jsonl" \
#     "ruler_8k" \
#     "answer_only" \
#     "logs/zero_shot_test_evals" \
#     50 \
#     16

# ============================================================================
# 3. GPT-OSS-20B Evaluations
# ============================================================================
# print_header "GPT-OSS-20B Evaluations"

# run_evaluation \
#     "gpt_oss_20b" \
#     "openai/gpt-oss-20b" \
#     "data/processed/math_test.csv" \
#     "math" \
#     "reasoning" \
#     "logs/zero_shot_test_evals" \
#     4096 \
#     16

# run_evaluation \
#     "gpt_oss_20b" \
#     "openai/gpt-oss-20b" \
#     "data/processed/ruler_8k_test.jsonl" \
#     "ruler_8k" \
#     "answer_only" \
#     "logs/zero_shot_test_evals" \
#     50 \
#     16

# ============================================================================
# 4. Qwen3-4B Evaluation (if using HuggingFace model)
# ============================================================================
# print_header "Qwen3-4B Evaluation"

# run_evaluation \
#     "qwen_4b" \
#     "Qwen/Qwen3-4B" \
#     "data/processed/math_test.csv" \
#     "math" \
#     "reasoning" \
#     "logs/zero_shot_test_evals" \
#     4096 \
#     32

# run_evaluation \
#     "qwen_4b" \
#     "Qwen/Qwen3-4B" \
#     "data/processed/ruler_4k_test.jsonl" \
#     "ruler_4k" \
#     "answer_only" \
#     "logs/zero_shot_test_evals" \
#     50 \
#     32

# ============================================================================
# Summary
# ============================================================================
# print_header "All evaluations completed!"

# echo -e "${GREEN}Test results saved in: logs/zero_shot_test_evals/${NC}"
# ls -lh logs/zero_shot_test_evals/*.jsonl 2>/dev/null || echo "No output files found"

# echo -e "\n${YELLOW}Summary of generated files:${NC}"
# find logs/zero_shot_test_evals -name "*_test.jsonl" -type f 2>/dev/null | sort | while read f; do
#     size=$(du -h "$f" | cut -f1)
#     echo "  - $(basename $f) ($size)"
# done

# echo ""