# #!/usr/bin/env bash
set -euo pipefail

TASK_TYPE=${TASK_TYPE:-"math"}

for seed in $(seq 0 9); do
  input_file="/mnt/home/chaeyun-jang/gcsft/logs/qwen4/math_seed_samples/train_seed_${seed}.jsonl"
  output_file="/mnt/home/chaeyun-jang/gcsft/logs/qwen4/math_seed_samples/outputs_seed_${seed}_parsed.jsonl"
  
  args=(
    --task_type "${TASK_TYPE}"
    --input_file "${input_file}"
    --output_file "${output_file}"
  )
  
  python -m experiments.parsing "${args[@]}"
  echo "Saved ${output_file}"
done

input_file="/mnt/home/chaeyun-jang/gcsft/logs/qwen4/math_seed_samples/outputs_base_argmax.jsonl"
output_file="/mnt/home/chaeyun-jang/gcsft/logs/qwen4/math_seed_samples/outputs_base_argmax_parsed.jsonl"

args=(
    --task_type "${TASK_TYPE}"
    --input_file "${input_file}"
    --output_file "${output_file}"
)

python -m experiments.parsing "${args[@]}"
echo "Saved ${output_file}"