#!/bin/bash

# GSM task - bs32*2 checkpoint
# input_file="/workspace/gcsft/logs/csft_test_evals/Llama-3.2-3B-Instruct_csft_single_gsm_seed0_lr0.0001_kl0.0_bs32*2_checkpoint-600/Llama-3.2-3B-Instruct_gsm_test.jsonl"
# output_file="/workspace/gcsft/logs/csft_test_evals/Llama-3.2-3B-Instruct_csft_single_gsm_seed0_lr0.0001_kl0.0_bs32*2_checkpoint-600/Llama-3.2-3B-Instruct_gsm_test_parsed.jsonl"

# args=(
#     --task_type "gsm"
#     --input_file "${input_file}"
#     --output_file "${output_file}"
# )

# python -m experiments.parsing "${args[@]}"
# echo "Saved ${output_file}"

# GSM task - prev checkpoint
input_file="/mnt/home/chaeyun-jang/gcsft/logs/zero_shot_test_evals/Llama-3.2-3B-Instruct_csft_multi_seed0_lr0.0001_kl1.0_bs1_gs32_ms2000_ck1_checkpoint-2000/Llama-3.2-3B-Instruct_gsm_test.jsonl"
output_file="/mnt/home/chaeyun-jang/gcsft/logs/zero_shot_test_evals/Llama-3.2-3B-Instruct_csft_multi_seed0_lr0.0001_kl1.0_bs1_gs32_ms2000_ck1_checkpoint-2000/Llama-3.2-3B-Instruct_gsm_test_parsed.jsonl"

args=(
    --task_type "gsm"
    --input_file "${input_file}"
    --output_file "${output_file}"
)

python -m experiments.parsing "${args[@]}"
echo "Saved ${output_file}"

# MATH task - bs32*2 checkpoint
input_file="/mnt/home/chaeyun-jang/gcsft/logs/zero_shot_test_evals/Llama-3.2-3B-Instruct_csft_multi_seed0_lr0.0001_kl1.0_checkpoint-2000/Llama-3.2-3B-Instruct_math_test.jsonl"
output_file="/mnt/home/chaeyun-jang/gcsft/logs/zero_shot_test_evals/Llama-3.2-3B-Instruct_csft_multi_seed0_lr0.0001_kl1.0_bs1_gs32_ms2000_ck1_checkpoint-2000/Llama-3.2-3B-Instruct_math_test_parsed.jsonl"

args=(
    --task_type "math"
    --input_file "${input_file}"
    --output_file "${output_file}"
)

# python -m experiments.parsing "${args[@]}"
# echo "Saved ${output_file}"

# # MATH task - prev checkpoint
# input_file="/workspace/gcsft/logs/csft_test_evals/Llama-3.2-3B-Instruct_csft_single_gsm_seed0_lr0.0001_kl0.0_prev_checkpoint-600/Llama-3.2-3B-Instruct_math_test.jsonl"
# output_file="/workspace/gcsft/logs/csft_test_evals/Llama-3.2-3B-Instruct_csft_single_gsm_seed0_lr0.0001_kl0.0_prev_checkpoint-600/Llama-3.2-3B-Instruct_math_test_parsed.jsonl"

# args=(
#     --task_type "math"
#     --input_file "${input_file}"
#     --output_file "${output_file}"
# )

# python -m experiments.parsing "${args[@]}"
# echo "Saved ${output_file}"