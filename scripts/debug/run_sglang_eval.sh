#!/usr/bin/env bash
set -euo pipefail

# Simple runner for eval_sglang_server.py assuming server is already running.
# Configure via env vars below or override inline before the command.
# Example:
#   EVAL_FILE=data/processed/hellaswag_val.csv \
#   OUTPUT_DIR=logs/sglang_eval \
#   DRY_RUN=1 \
#   bash scripts/run_sglang_eval.sh

HOST=${HOST:-localhost}
PORT=${PORT:-30000}
SEED=${SEED:-0}
EVAL_FILE=${EVAL_FILE:-data/processed/hellaswag_val.csv}
OUTPUT_DIR=${OUTPUT_DIR:-logs/sglang_eval}
OUTPUT_FILE=${OUTPUT_FILE:-}
SYSTEM_PROMPT=${SYSTEM_PROMPT:-}
ADD_CONF=${ADD_CONF:-0}              # 1 to enable confidence scoring
CONF_PROMPT=${CONF_PROMPT:-default}
INCLUDE_PROMPT=${INCLUDE_PROMPT:-0}  # 1 to include prompts in outputs
DRY_RUN=${DRY_RUN:-0}                # 1 to skip server requests
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
TEMPERATURE=${TEMPERATURE:-0.7}
TOP_P=${TOP_P:-0.9}

mkdir -p "${OUTPUT_DIR}"

args=(
  --host "${HOST}"
  --port "${PORT}"
  --seed "${SEED}"
  --eval_file "${EVAL_FILE}"
  --output_dir "${OUTPUT_DIR}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --temperature "${TEMPERATURE}"
  --top_p "${TOP_P}"
)

if [[ -n "${OUTPUT_FILE}" ]]; then
  args+=(--output_file "${OUTPUT_FILE}")
fi
if [[ -n "${SYSTEM_PROMPT}" ]]; then
  args+=(--system_prompt "${SYSTEM_PROMPT}")
fi
if [[ "${ADD_CONF}" == "1" ]]; then
  args+=(--add_conf --confidence_prompt_name "${CONF_PROMPT}")
fi
if [[ "${INCLUDE_PROMPT}" == "1" ]]; then
  args+=(--include_prompt)
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  args+=(--dry_run)
fi

python3 experiments/eval_sglang_server.py "${args[@]}"
