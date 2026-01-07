#!/usr/bin/env python
import argparse
import json
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.distributions import Categorical, kl_divergence
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
    parser.add_argument("--prompt_key", default="prompt")
    parser.add_argument("--response_key", default="response")
    parser.add_argument("--confidence_key", default="confidence")
    parser.add_argument("--reference_key", default="reference")
    parser.add_argument("--kl_decay", type=float, default=0.0)
    parser.add_argument("--kl_type", default="reverse_kl", choices=["reverse_kl", "forward_kl", "jsd"])
    parser.add_argument("--ref_adapter_name", default="ref")
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


def build_text(
    example: Dict[str, Any],
    tokenizer: AutoTokenizer,
    prompt_key: str,
    response_key: str,
) -> str:
    if "messages" in example:
        return tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
    if prompt_key in example and response_key in example:
        return f"{example[prompt_key]}\n{example[response_key]}"
    if response_key in example:
        return example[response_key]
    if "text" in example:
        return example["text"]
    raise ValueError(
        "Example must contain 'messages', prompt/response, response, or 'text'."
    )


def select_response_key(
    example: Dict[str, Any],
    preferred_key: str,
    fallback_key: str,
) -> Optional[str]:
    if preferred_key in example:
        return preferred_key
    if fallback_key in example:
        return fallback_key
    if "text" in example:
        return "text"
    if "messages" in example:
        return "messages"
    return None


class ConfidenceDataCollator:
    def __init__(self, tokenizer: AutoTokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _tokenize(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = tokenized["input_ids"].clone()
        labels[tokenized["attention_mask"] == 0] = -100
        tokenized["labels"] = labels
        return tokenized

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        conf_texts = [example["conf_text"] for example in batch]
        conf_batch = self._tokenize(conf_texts)

        output = {
            "conf_input_ids": conf_batch["input_ids"],
            "conf_attention_mask": conf_batch["attention_mask"],
            "conf_labels": conf_batch["labels"],
        }

        ref_texts = [example.get("ref_text") for example in batch]
        if all(text is not None for text in ref_texts):
            ref_batch = self._tokenize(ref_texts)  # type: ignore[arg-type]
            output.update(
                {
                    "ref_input_ids": ref_batch["input_ids"],
                    "ref_attention_mask": ref_batch["attention_mask"],
                    "ref_labels": ref_batch["labels"],
                }
            )

        return output


def prepare_dataset(dataset: Dataset, tokenizer: AutoTokenizer, args: argparse.Namespace) -> Dataset:
    def _map(example: Dict[str, Any]) -> Dict[str, Any]:
        conf_key = select_response_key(example, args.confidence_key, args.response_key)
        ref_key = select_response_key(example, args.reference_key, args.response_key)
        if conf_key is None:
            raise ValueError("Could not resolve confidence response field for example.")
        conf_text = build_text(example, tokenizer, args.prompt_key, conf_key)
        ref_text = None
        if ref_key is not None:
            ref_text = build_text(example, tokenizer, args.prompt_key, ref_key)
        return {"conf_text": conf_text, "ref_text": ref_text}

    return dataset.map(_map, remove_columns=dataset.column_names)


class ConfidenceSFTTrainer(SFTTrainer):
    def __init__(
        self,
        *args,
        kl_decay: float = 0.0,
        kl_type: str = "reverse_kl",
        ref_adapter_name: str = "ref",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.kl_decay = kl_decay
        self.kl_type = kl_type
        self.ref_adapter_name = ref_adapter_name

    def compute_kl_loss(self, model, ref_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.kl_decay <= 0.0:
            return torch.tensor(0.0, device=self.accelerator.device)

        if not hasattr(model, "set_adapter"):
            raise ValueError("KL loss requires a PEFT model with adapters.")

        ref_inputs = {k: v.to(self.accelerator.device) for k, v in ref_inputs.items()}

        probs = model(**ref_inputs).logits[..., :-1, :].softmax(dim=-1)
        with torch.inference_mode():
            model.set_adapter(self.ref_adapter_name)
            ref_probs = model(**ref_inputs).logits[..., :-1, :].softmax(dim=-1)
            model.set_adapter("default")

        model.train()
        labels = ref_inputs.pop("labels")[..., 1:]

        p = Categorical(probs=probs)
        p_ref = Categorical(probs=ref_probs)

        if self.kl_type == "reverse_kl":
            kl_loss = kl_divergence(p, p_ref)
        elif self.kl_type == "forward_kl":
            kl_loss = kl_divergence(p_ref, p)
        elif self.kl_type == "jsd":
            p_mix = Categorical(probs=(probs + ref_probs) / 2)
            kl_loss = (kl_divergence(p, p_mix) + kl_divergence(p_ref, p_mix)) / 2
        else:
            raise NotImplementedError

        loss_mask = labels != -100
        loss = (kl_loss * loss_mask).sum(dim=-1).mean(dim=0)

        return loss

    def _collate_fn(
        self, inputs: Dict[str, torch.Tensor], targets_key: str
    ) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": inputs[f"{targets_key}_input_ids"],
            "attention_mask": inputs[f"{targets_key}_attention_mask"],
            "labels": inputs[f"{targets_key}_labels"],
        }

    def compute_loss(self, model, inputs, return_outputs=False):
        conf_inputs = self._collate_fn(inputs, "conf")
        conf_outputs = model(**conf_inputs)
        conf_loss = conf_outputs.loss

        kl_loss = torch.tensor(0.0, device=conf_loss.device)
        if self.kl_decay > 0.0 and "ref_input_ids" in inputs:
            ref_inputs = self._collate_fn(inputs, "ref")
            kl_loss = self.compute_kl_loss(model, ref_inputs)

        loss = conf_loss + self.kl_decay * kl_loss
        return (loss, conf_outputs) if return_outputs else loss


def main() -> None:
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = args.tf32

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = prepare_dataset(load_any_dataset(args.train_file), tokenizer, args)
    eval_ds = None
    if args.validation_file:
        eval_ds = prepare_dataset(load_any_dataset(args.validation_file), tokenizer, args)

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

    data_collator = ConfidenceDataCollator(tokenizer=tokenizer, max_length=args.max_seq_length)

    trainer = ConfidenceSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        kl_decay=args.kl_decay,
        kl_type=args.kl_type,
        ref_adapter_name=args.ref_adapter_name,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
