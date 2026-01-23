#!/bin/bash

python -m experiments.train.train_csft \
  --model_name meta-llama/Llama-3.2-3B-Instruct \
  --train_type multi \
  --batch_size 1 \
  --gradient_accumulation_steps 32 \
  --learning_rate 1e-04 \
  --use_lora \
  --confidence_key conf_label_single \
  --confidence_input_key conf_input_single \
  --kl_decay 1.0 