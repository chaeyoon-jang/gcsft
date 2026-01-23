#!/bin/bash

# GRPO Training Script for Llama-3.2-3B-Instruct
# Usage: bash scripts/train_grpo_llama.sh [brier|log_loss] [train_type]

set -e

# Arguments
REWARD_MODE=${1:-"brier"}  # brier or log_loss
TRAIN_TYPE=${2:-"gsm"}  

# Model configuration
MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"
LOG_DIR="logs/grpo_experiments"

# Data configuration
CONFIDENCE_INPUT_KEY="conf_input"
CONFIDENCE_KEY="tf"

# Generation parameters
MAX_SEQ_LENGTH=4096
MAX_PROMPT_LENGTH=1024
MAX_COMPLETION_LENGTH=50
TEMPERATURE=0.7
TOP_P=0.9

# Training parameters
BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=32
NUM_TRAIN_EPOCHS=1
MAX_STEPS=2000
LEARNING_RATE=1e-4
KL_DECAY=0.0
SEED=42

# LoRA configuration
USE_LORA="--use_lora"

# Reward function
LOG_LOSS_EPSILON=1e-4

# Training settings
BF16="--bf16"
GRADIENT_CHECKPOINTING="--gradient_checkpointing"

# Logging
USE_WANDB=""  # Add "--use_wandb" to enable W&B logging

# VLLM (for faster generation)
USE_VLLM=""  # Add "--use_vllm" to enable vLLM
VLLM_DEVICE="auto"
VLLM_GPU_MEMORY_UTILIZATION=0.9

echo "=========================================="
echo "Starting GRPO Training"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Reward Mode: $REWARD_MODE"
echo "Train Type: $TRAIN_TYPE"
echo "Learning Rate: $LEARNING_RATE"
echo "Batch Size: $BATCH_SIZE"
echo "Gradient Accumulation Steps: $GRADIENT_ACCUMULATION_STEPS"
echo "Max Steps: $MAX_STEPS"
echo "=========================================="

python experiments/train/train_grpo.py \
    --model_name "$MODEL_NAME" \
    --train_type "$TRAIN_TYPE" \
    --log_dir "$LOG_DIR" \
    --confidence_input_key "$CONFIDENCE_INPUT_KEY" \
    --confidence_key "$CONFIDENCE_KEY" \
    --max_seq_length $MAX_SEQ_LENGTH \
    --max_prompt_length $MAX_PROMPT_LENGTH \
    --max_completion_length $MAX_COMPLETION_LENGTH \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --max_steps $MAX_STEPS \
    --learning_rate $LEARNING_RATE \
    --kl_decay $KL_DECAY \
    --seed $SEED \
    $BF16 \
    $GRADIENT_CHECKPOINTING \
    $USE_LORA \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --lora_target_modules $LORA_TARGET_MODULES \
    --reward_mode "$REWARD_MODE" \
    --reward_format_pattern "$REWARD_FORMAT_PATTERN" \
    --log_loss_epsilon $LOG_LOSS_EPSILON \
    $USE_WANDB \
    $USE_VLLM \
    --vllm_device "$VLLM_DEVICE" \
    --vllm_gpu_memory_utilization $VLLM_GPU_MEMORY_UTILIZATION

echo "=========================================="
echo "Training completed!"
echo "Output directory: $LOG_DIR"
echo "=========================================="
