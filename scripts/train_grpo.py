#!/usr/bin/env python
import argparse
import json
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import GRPOConfig, GRPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO fine-tuning with TRL")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--reward_model")
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_completion_length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
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


def build_prompt(example: Dict[str, Any], tokenizer: AutoTokenizer) -> str:
    if "prompt" in example:
        return example["prompt"]
    if "messages" in example:
        return tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=True
        )
    if "text" in example:
        return example["text"]
    raise ValueError("Example must contain 'prompt', 'messages', or 'text'.")


def extract_reference(example: Dict[str, Any]) -> Optional[str]:
    if "reference" in example:
        return example["reference"]
    if "response" in example:
        return example["response"]
    if "answer" in example:
        return example["answer"]
    return None


def prepare_dataset(dataset: Dataset, tokenizer: AutoTokenizer) -> Dataset:
    def _map(example: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "prompt": build_prompt(example, tokenizer),
            "reference": extract_reference(example),
        }

    return dataset.map(_map, remove_columns=dataset.column_names)


def build_reward_model(reward_model_name: str, bf16: bool):
    tokenizer = AutoTokenizer.from_pretrained(reward_model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_name,
        torch_dtype=torch.bfloat16 if bf16 else torch.float16,
    )
    model.eval()
    return tokenizer, model


def main() -> None:
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = args.tf32

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_dataset = load_any_dataset(args.train_file)
    train_dataset = prepare_dataset(raw_dataset, tokenizer)

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

    reward_tokenizer = None
    reward_model = None
    if args.reward_model:
        reward_tokenizer, reward_model = build_reward_model(args.reward_model, args.bf16)

    def heuristic_reward(prompts: List[str], completions: List[str], **kwargs: Any) -> List[float]:
        rewards: List[float] = []
        references = kwargs.get("reference")
        for idx, completion in enumerate(completions):
            ref = None
            if references is not None and idx < len(references):
                ref = references[idx]
            if ref:
                rewards.append(1.0 if ref.strip().lower() in completion.strip().lower() else 0.0)
            else:
                rewards.append(min(len(completion) / 200.0, 1.0))
        return rewards

    def reward_model_fn(prompts: List[str], completions: List[str], **kwargs: Any) -> List[float]:
        if reward_model is None or reward_tokenizer is None:
            return heuristic_reward(prompts, completions, **kwargs)
        inputs = reward_tokenizer(
            [f"{p}{c}" for p, c in zip(prompts, completions)],
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(reward_model.device)
        with torch.no_grad():
            scores = reward_model(**inputs).logits.squeeze(-1)
        return scores.detach().float().tolist()

    config = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        optim="adamw_torch_fused",
        lr_scheduler_type="cosine",
        seed=args.seed,
        report_to=[],
    )

    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        reward_funcs=[reward_model_fn],
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
