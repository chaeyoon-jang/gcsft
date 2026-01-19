#!/bin/bash

# GSM task
input_file="/mnt/home/chaeyun-jang/gcsft/logs/zero_shot_test_evals/llama_3b_gsm_test.jsonl"
output_file="/mnt/home/chaeyun-jang/gcsft/logs/zero_shot_test_evals/llama_3b_gsm_test_parsed.jsonl"

args=(
    --task_type "gsm"
    --input_file "${input_file}"
    --output_file "${output_file}"
)

python -m experiments.parsing "${args[@]}"
echo "Saved ${output_file}"

# # MATH task (Qwen 4B)
# input_file="/mnt/home/chaeyun-jang/gcsft/logs/zero_shot_test_evals/qwen_4b_math_test.jsonl"
# output_file="/mnt/home/chaeyun-jang/gcsft/logs/zero_shot_test_evals/qwen_4b_math_test_parsed.jsonl"

# args=(
#     --task_type "math"
#     --input_file "${input_file}"
#     --output_file "${output_file}"
# )

# python -m experiments.parsing "${args[@]}"
# echo "Saved ${output_file}"

# # MATH task (Qwen 8B)
# input_file="/mnt/home/chaeyun-jang/gcsft/logs/zero_shot_test_evals/qwen_8b_math_test.jsonl"
# output_file="/mnt/home/chaeyun-jang/gcsft/logs/zero_shot_test_evals/qwen_8b_math_test_parsed.jsonl"

# args=(
#     --task_type "math"
#     --input_file "${input_file}"
#     --output_file "${output_file}"
# )

# python -m experiments.parsing "${args[@]}"
# echo "Saved ${output_file}"