#!/usr/bin/env python
import argparse
from dataclasses import fields
from pathlib import Path
from typing import Any, List

import pandas as pd 
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import GRPOConfig, GRPOTrainer

from utils.reward_func import brier_reward, log_loss_reward, strict_confidence_reward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO fine-tuning with TRL")
    
    # Model and data
    parser.add_argument("--model_name", default="meta-llama/Llama-3.2-3B-Instruct",help="Model name or path")
    parser.add_argument("--train_type", default="gsm", help="Training type (e.g., gsm)")
    parser.add_argument("--log_dir", default="logs/grpo_experiments", help="Directory for logs and checkpoints")
    
    # Data columns
    parser.add_argument("--confidence_input_key", default="prompt", help="Column name for prompts")
    parser.add_argument("--confidence_key", default="tf", help="Column name for correctness (0 or 1)")
    
    # Generation parameters
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--max_prompt_length", type=int, default=3000)
    parser.add_argument("--max_completion_length", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=1, help="Per device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--kl_decay", type=float, default=1.0, help="KL divergence weight decay")
    parser.add_argument("--seed", type=int, default=42)
    
    # Model settings
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    
    # LoRA
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_target_modules", nargs="*", default=["q_proj", "v_proj"])
    parser.add_argument("--ref_adapter_name", default="reference", help="Reference adapter name for KL")
    
    # Reward function
    parser.add_argument(
        "--reward_mode",
        choices=["brier", "log_loss", "strict_conf"],
        default="brier",
        help="Reward function type",
    )
    parser.add_argument("--log_loss_epsilon", type=float, default=1e-4, help="Epsilon for log-loss to prevent -inf")
    
    # Other
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--vllm_device", default="auto")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--deepspeed", default=None)
    parser.add_argument("--debug_print_batches", type=int, default=0, help="Print first N reward batches for debugging")
    parser.add_argument("--debug_print_interval", type=int, default=0, help="Print every K batches (after the first N if set)")
    
    return parser.parse_args()


def load_datasets(args):
    print("Loading datasets...")
    
    model_name_short = args.model_name.split("/")[-1]
    base_path = Path("data/train_data") / model_name_short / "rl_base"
    
    train_df = pd.read_csv(base_path / f"{args.train_type}_train.csv").dropna()
    eval_df = pd.read_csv(base_path / f"{args.train_type}_valid.csv").dropna()
        
    train_df = train_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    eval_df = eval_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    
    train_ds = Dataset.from_pandas(train_df)
    eval_ds = Dataset.from_pandas(eval_df)
    
    print(f"Loaded {len(train_ds)} training samples and {len(eval_ds)} eval samples")
    return train_ds, eval_ds



def load_model_and_tokenizer(args):
    print(f"Loading model: {args.model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        padding_side="left",
        model_max_length=args.max_seq_length,
        use_fast=True,
    )
    # Keep the end of the prompt (contains question + confidence instruction) if truncation happens
    tokenizer.truncation_side = "left"
    
    # Set pad token - use EOS token to avoid out-of-bounds issues
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    torch_dtype = torch.float16 if not torch.cuda.is_bf16_supported() else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    
    model.config.pad_token_id = tokenizer.pad_token_id
    print("Model loaded successfully.")
    
    return model, tokenizer


def setup_lora(args, model):
    if not args.use_lora:
        return model
    
    print(f"Setting up LoRA (r={args.lora_rank}, alpha={args.lora_alpha})")
    
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    if args.kl_decay > 0.0:
        model.add_adapter(args.ref_adapter_name, lora_config)
    model.set_adapter("default")
    
    return model


def main(args):
    
    set_seed(args.seed)
    
    sub_dir = f"{args.model_name.split('/')[-1]}_{args.reward_mode}_{args.train_type}_seed{args.seed}_lr{args.learning_rate}_kl{args.kl_decay}_bs{args.batch_size}_gs{args.gradient_accumulation_steps}_ms{args.max_steps}"
    output_dir = Path(args.log_dir) / sub_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")

    train_ds, eval_ds = load_datasets(args)
    model, tokenizer = load_model_and_tokenizer(args)

    # Filter out prompts that would exceed max_prompt_length to avoid truncation removing instructions
    def _len_ok(example):
        return len(tokenizer.encode(example[args.confidence_input_key], add_special_tokens=False)) <= args.max_prompt_length

    before_train = len(train_ds)
    before_eval = len(eval_ds)
    train_ds = train_ds.filter(_len_ok)
    eval_ds = eval_ds.filter(_len_ok)
    print(f"Filtered long prompts: train {before_train} -> {len(train_ds)}, eval {before_eval} -> {len(eval_ds)}")
    cols_to_keep = [       
        args.confidence_input_key,
        args.confidence_key,
    ]
    cols_to_remove = [col for col in train_ds.column_names if col not in cols_to_keep]
    
    if cols_to_remove:
        train_ds = train_ds.remove_columns(cols_to_remove)
        eval_ds = eval_ds.remove_columns(cols_to_remove)
    
    print(f"Dataset columns: {train_ds.column_names}")

    if args.use_lora:
        model = setup_lora(args, model)

    config_kwargs = {
        "seed": args.seed,
        "output_dir": str(output_dir),
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "bf16": args.bf16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "optim": "adamw_torch_fused",
        "lr_scheduler_type": "cosine",
        "logging_steps": 10,
        "save_steps": 200,
        "save_total_limit": 3,
        "report_to": ["wandb"] if args.use_wandb else [],
        "use_vllm": args.use_vllm,
        "vllm_device": args.vllm_device,
        "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "deepspeed": args.deepspeed,
        "scale_rewards": False, 
        "loss_type": "dr_grpo",  
        #"chat_template_kwargs": {},  # Disable chat template - prompts are already formatted
        "beta": args.kl_decay,  # KL divergence penalty weight
    }
    allowed_keys = {field.name for field in fields(GRPOConfig)}
    config = GRPOConfig(**{key: value for key, value in config_kwargs.items() if key in allowed_keys})

    debug_counter = {"batches": 0}

    def _maybe_debug_print(prompts: List[str], completions: List[str], rewards: List[float]):
        debug_counter["batches"] += 1
        batch_idx = debug_counter["batches"]

        should_print = False
        if args.debug_print_batches > 0 and batch_idx <= args.debug_print_batches:
            should_print = True
        elif args.debug_print_interval > 0 and batch_idx % args.debug_print_interval == 0:
            should_print = True

        if not should_print:
            return

        sample_prompt = prompts[0] if prompts else ""
        sample_completion = completions[0] if completions else ""
        print("\n[DEBUG reward] batch", batch_idx)
        print("prompt[0]:", sample_prompt)
        print("completion[0]:", sample_completion)
        print("rewards (first 8):", rewards[:8])

    def brier_reward_fn(prompts: List[str], completions: List[str], **kwargs: Any) -> List[float]:
        rewards = brier_reward(
            completions,
            kwargs.get("tf") or [],
        )
        _maybe_debug_print(prompts, completions, rewards)
        return rewards

    def log_loss_reward_fn(
        prompts: List[str], completions: List[str], **kwargs: Any
    ) -> List[float]:
        rewards = log_loss_reward(
            completions,
            kwargs.get("tf") or [],
            epsilon=args.log_loss_epsilon,
        )
        _maybe_debug_print(prompts, completions, rewards)
        return rewards

    def strict_conf_reward_fn(prompts: List[str], completions: List[str], **kwargs: Any) -> List[float]:
        rewards = strict_confidence_reward(completions)
        _maybe_debug_print(prompts, completions, rewards)
        return rewards

    if args.reward_mode == "brier":
        reward_funcs = [brier_reward_fn] #, strict_conf_reward_fn]
        print("Using Brier score reward")
    elif args.reward_mode == "log_loss":
        reward_funcs = [log_loss_reward_fn]
        print("Using Log-loss reward")
    elif args.reward_mode == "strict_conf":
        reward_funcs = [strict_conf_reward_fn]
        print("Using strict confidence format reward")
    else:
        raise ValueError(f"Unknown reward mode: {args.reward_mode}")

    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=train_ds,
        reward_funcs=reward_funcs,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(output_dir))


if __name__ == "__main__":
    args = parse_args()
    main(args)
