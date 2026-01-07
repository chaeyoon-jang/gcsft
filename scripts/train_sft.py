#!/usr/bin/env python
import argparse
import json
from typing import Any, Dict, List

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT fine-tuning with TRL")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--validation_file")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--packing", action="store_true")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--use_flash_attn", action="store_true")
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", nargs="*", default=None)
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--use_8bit", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_jsonl(path: str) -> Dataset:
    with open(path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    return Dataset.from_list(records)


def load_any_dataset(path: str) -> Dataset:
    if path.endswith(".jsonl"):
        return load_jsonl(path)
    if path.endswith(".json"):
        return load_dataset("json", data_files=path, split="train")
    return load_dataset(path, split="train")


def ensure_prompt_fields(example: Dict[str, Any]) -> Dict[str, Any]:
    if "text" in example:
        return {"text": example["text"]}
    if "prompt" in example and "response" in example:
        return {"text": f"{example['prompt']}\n{example['response']}"}
    if "messages" in example:
        return {"messages": example["messages"]}
    raise ValueError("Example must contain 'text', 'prompt'+'response', or 'messages'.")


def build_text(example: Dict[str, Any], tokenizer: AutoTokenizer) -> str:
    if "text" in example:
        return example["text"]
    if "messages" in example:
        return tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
    if "prompt" in example and "response" in example:
        return f"{example['prompt']}\n{example['response']}"
    raise ValueError("Unsupported example format.")


def main() -> None:
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = args.tf32

    train_ds = load_any_dataset(args.train_file).map(ensure_prompt_fields)
    eval_ds = None
    if args.validation_file:
        eval_ds = load_any_dataset(args.validation_file).map(ensure_prompt_fields)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if args.use_4bit or args.use_8bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=args.use_4bit,
            load_in_8bit=args.use_8bit,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        quantization_config=quant_config,
        attn_implementation="flash_attention_2" if args.use_flash_attn else None,
    )

    if args.use_4bit or args.use_8bit:
        model = prepare_model_for_kbit_training(model)

    if args.lora:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        evaluation_strategy="steps" if eval_ds is not None else "no",
        save_strategy="steps",
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        optim="adamw_torch_fused",
        lr_scheduler_type="cosine",
        seed=args.seed,
        report_to=[],
    )

    def formatting_func(example: Dict[str, Any]) -> List[str]:
        return [build_text(example, tokenizer)]

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        formatting_func=formatting_func,
        max_seq_length=args.max_seq_length,
        packing=args.packing,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
