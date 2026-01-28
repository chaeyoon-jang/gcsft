python -m experiments.train.train_csft \
  --model_name meta-llama/Llama-3.2-3B-Instruct \
  --train_type multi \
  --batch_size 1 \
  --gradient_accumulation_steps 64 \
  --learning_rate 1e-05 \
  --use_lora \
  --confidence_key conf_label_multi \
  --confidence_input_key conf_input_multi \
  --kl_decay 0.0 \
  --use_wandb \
  --max_steps 5000 

python -m experiments.train.train_csft \
  --model_name meta-llama/Llama-3.2-3B-Instruct \
  --train_type multi \
  --batch_size 1 \
  --gradient_accumulation_steps 64 \
  --learning_rate 1e-06 \
  --use_lora \
  --confidence_key conf_label_multi \
  --confidence_input_key conf_input_multi \
  --kl_decay 0.0 \
  --use_wandb \
  --max_steps 5000 


python -m experiments.train.train_csft \
  --model_name meta-llama/Llama-3.2-3B-Instruct \
  --train_type multi \
  --batch_size 1 \
  --gradient_accumulation_steps 64 \
  --learning_rate 1e-4 \
  --use_lora \
  --confidence_key conf_label_multi \
  --confidence_input_key conf_input_multi \
  --kl_decay 0.0 \
  --use_wandb \
  --max_steps 5000 