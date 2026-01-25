input_file="/mnt/home/chaeyun-jang/gcsft/logs/single_finals/Llama-3.2-3B-Instruct_csft_single_gsm_seed0_lr0.0001_kl0.0_bs2_gs16_ms2000_ck0_checkpoint-600/Llama-3.2-3B-Instruct_gsm_test.jsonl"
output_file="/mnt/home/chaeyun-jang/gcsft/logs/single_finals/Llama-3.2-3B-Instruct_csft_single_gsm_seed0_lr0.0001_kl0.0_bs2_gs16_ms2000_ck0_checkpoint-600/Llama-3.2-3B-Instruct_gsm_test_parsed.jsonl"

args=(
    --task_type "gsm"
    --input_file "${input_file}"
    --output_file "${output_file}"
)

python -m experiments.parsing "${args[@]}"
echo "Saved ${output_file}"

###
input_file="/mnt/home/chaeyun-jang/gcsft/logs/single_finals/Llama-3.2-3B-Instruct_csft_single_gsm_seed0_lr0.0001_kl0.0_bs2_gs16_ms2000_ck1_checkpoint-800/Llama-3.2-3B-Instruct_gsm_test.jsonl"
output_file="/mnt/home/chaeyun-jang/gcsft/logs/single_finals/Llama-3.2-3B-Instruct_csft_single_gsm_seed0_lr0.0001_kl0.0_bs2_gs16_ms2000_ck1_checkpoint-800/Llama-3.2-3B-Instruct_gsm_test_parsed.jsonl"

args=(
    --task_type "gsm"
    --input_file "${input_file}"
    --output_file "${output_file}"
)

python -m experiments.parsing "${args[@]}"
echo "Saved ${output_file}"

###
input_file="/mnt/home/chaeyun-jang/gcsft/logs/single_finals/Llama-3.2-3B-Instruct_csft_single_ruler_4k_seed0_lr0.0001_kl0.0_bs2_gs16_ms2000_ck0_checkpoint-800/Llama-3.2-3B-Instruct_gsm_test.jsonl"
output_file="/mnt/home/chaeyun-jang/gcsft/logs/single_finals/Llama-3.2-3B-Instruct_csft_single_ruler_4k_seed0_lr0.0001_kl0.0_bs2_gs16_ms2000_ck0_checkpoint-800/Llama-3.2-3B-Instruct_gsm_test_parsed.jsonl"

args=(
    --task_type "gsm"
    --input_file "${input_file}"
    --output_file "${output_file}"
)

python -m experiments.parsing "${args[@]}"
echo "Saved ${output_file}"

input_file="/mnt/home/chaeyun-jang/gcsft/logs/single_finals/Llama-3.2-3B-Instruct_csft_single_gsm_seed0_lr0.0001_kl0.0_bs2_gs16_ms2000_ck0_checkpoint-600/Llama-3.2-3B-Instruct_math_test.jsonl"
output_file="/mnt/home/chaeyun-jang/gcsft/logs/single_finals/Llama-3.2-3B-Instruct_csft_single_gsm_seed0_lr0.0001_kl0.0_bs2_gs16_ms2000_ck0_checkpoint-600/Llama-3.2-3B-Instruct_math_test_parsed.jsonl"

args=(
    --task_type "math"
    --input_file "${input_file}"
    --output_file "${output_file}"
)

python -m experiments.parsing "${args[@]}"
echo "Saved ${output_file}"

###
input_file="/mnt/home/chaeyun-jang/gcsft/logs/single_finals/Llama-3.2-3B-Instruct_csft_single_gsm_seed0_lr0.0001_kl0.0_bs2_gs16_ms2000_ck1_checkpoint-800/Llama-3.2-3B-Instruct_math_test.jsonl"
output_file="/mnt/home/chaeyun-jang/gcsft/logs/single_finals/Llama-3.2-3B-Instruct_csft_single_gsm_seed0_lr0.0001_kl0.0_bs2_gs16_ms2000_ck1_checkpoint-800/Llama-3.2-3B-Instruct_math_test_parsed.jsonl"

args=(
    --task_type "math"
    --input_file "${input_file}"
    --output_file "${output_file}"
)

python -m experiments.parsing "${args[@]}"
echo "Saved ${output_file}"

###
input_file="/mnt/home/chaeyun-jang/gcsft/logs/single_finals/Llama-3.2-3B-Instruct_csft_single_ruler_4k_seed0_lr0.0001_kl0.0_bs2_gs16_ms2000_ck0_checkpoint-800/Llama-3.2-3B-Instruct_math_test.jsonl"
output_file="/mnt/home/chaeyun-jang/gcsft/logs/single_finals/Llama-3.2-3B-Instruct_csft_single_ruler_4k_seed0_lr0.0001_kl0.0_bs2_gs16_ms2000_ck0_checkpoint-800/Llama-3.2-3B-Instruct_math_test_parsed.jsonl"

args=(
    --task_type "math"
    --input_file "${input_file}"
    --output_file "${output_file}"
)

python -m experiments.parsing "${args[@]}"
echo "Saved ${output_file}"