#!/usr/bin/env python
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import wandb
from datasets import Dataset, load_dataset
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from transformers import DataCollatorWithPadding

from utils.calibration_utils import ece, brier_score, nll, auroc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probing: train a head on top of frozen LLM")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--validation_file")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--use_flash_attn", action="store_true")
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--use_8bit", action="store_true")
    parser.add_argument("--log_dir", default=None)
    parser.add_argument("--prompt_key", default="prompt")
    parser.add_argument("--response_key", default="response")
    parser.add_argument("--confidence_key", default="confidence")
    parser.add_argument("--reference_key", default="reference")
    parser.add_argument("--target_type", choices=["confidence", "correctness"], default="confidence")
    parser.add_argument("--pooling", choices=["last", "mean"], default="last")
    parser.add_argument("--head_hidden_dim", type=int, default=0, help="0 for linear head; >0 for 1-hidden-layer MLP")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_run_name", default=None)
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


def resolve_log_dir(log_dir: Optional[str]) -> Path:
    base_dir = Path("logs")
    base_dir.mkdir(parents=True, exist_ok=True)
    if log_dir:
        run_dir = base_dir / log_dir
    else:
        run_dir = base_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


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


def normalize_answer(s: str) -> str:
    import re, string
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match(pred: str, ref: str) -> int:
    return int(normalize_answer(pred) == normalize_answer(ref))


def prepare_probe_dataset(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    args: argparse.Namespace,
) -> Tuple[Dataset, str]:
    """Tokenize and produce labels for probing.

    Returns the processed dataset and the name of the label field ("labels").
    """
    def _map(example: Dict[str, Any]) -> Dict[str, Any]:
        # build text (prompt + response if available)
        text = build_text(example, tokenizer, args.prompt_key, args.response_key)
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=args.max_seq_length,
        )

        if args.target_type == "confidence":
            if args.confidence_key not in example:
                raise ValueError("Confidence target not found in example.")
            label = float(example[args.confidence_key])
            label = max(0.0, min(label, 1.0))
        else:  # correctness
            if args.response_key not in example or args.reference_key not in example:
                raise ValueError("Correctness target needs response and reference.")
            label = exact_match(str(example[args.response_key]), str(example[args.reference_key]))

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": label,
        }

    processed = dataset.map(_map, remove_columns=dataset.column_names)
    return processed, "labels"


class ProbeHead(nn.Module):
    def __init__(self, hidden_size: int, head_hidden_dim: int = 0, dropout: float = 0.0, target_type: str = "confidence"):
        super().__init__()
        self.target_type = target_type
        if head_hidden_dim and head_hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, head_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden_dim, 1),
            )
        else:
            self.net = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1),
            )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # returns raw output: for confidence use sigmoid later; for correctness use logits
        return self.net(h).squeeze(-1)


def get_last_hidden(hidden_states: torch.Tensor, attention_mask: torch.Tensor, pooling: str = "last") -> torch.Tensor:
    # hidden_states: [B, T, H], attention_mask: [B, T]
    if pooling == "mean":
        lengths = attention_mask.sum(dim=1).clamp(min=1).unsqueeze(1)  # [B,1]
        masked = hidden_states * attention_mask.unsqueeze(-1)
        return masked.sum(dim=1) / lengths
    # last non-pad token
    lengths = attention_mask.sum(dim=1) - 1  # [B]
    idx = lengths.clamp(min=0)
    return hidden_states[torch.arange(hidden_states.size(0), device=hidden_states.device), idx]


@torch.no_grad()
def evaluate(
    base_model: AutoModelForCausalLM,
    head: ProbeHead,
    dataloader: DataLoader,
    pooling: str,
    target_type: str,
) -> Dict[str, float]:
    base_model.eval()
    head.eval()
    device = next(head.parameters()).device
    all_labels: List[float] = []
    all_preds: List[float] = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
        hs = outputs.hidden_states[-1]  # last layer
        h_last = get_last_hidden(hs, attention_mask, pooling)
        logits = head(h_last)

        labels = batch["labels"].detach().cpu().numpy().astype(np.float32)
        if target_type == "confidence":
            preds = torch.sigmoid(logits).detach().cpu().numpy()
        else:
            preds = torch.sigmoid(logits).detach().cpu().numpy()

        all_labels.extend(labels.tolist())
        all_preds.extend(preds.tolist())

    # compute metrics
    labels_np = np.array(all_labels)
    preds_np = np.array(all_preds)
    if target_type == "confidence":
        mse = float(np.mean((preds_np - labels_np) ** 2))
        mae = float(np.mean(np.abs(preds_np - labels_np)))
        return {"mse": mse, "mae": mae}
    else:
        # correctness calibration
        try:
            metrics = {
                "ece": float(ece(labels_np, preds_np, n_bins=10)),
                "brier": float(brier_score(labels_np, preds_np)),
                "nll": float(nll(labels_np, preds_np)),
                "auroc": float(auroc(labels_np, preds_np)),
            }
        except Exception:
            metrics = {
                "ece": float(ece(labels_np, preds_np, n_bins=10)),
                "brier": float(brier_score(labels_np, preds_np)),
                "nll": float(nll(labels_np, preds_np)),
                "auroc": float("nan"),
            }
        return metrics


def main() -> None:
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    log_dir = resolve_log_dir(args.log_dir)
    output_dir = log_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_raw = load_any_dataset(args.train_file)
    val_raw = load_any_dataset(args.validation_file) if args.validation_file else None

    train_ds, label_field = prepare_probe_dataset(train_raw, tokenizer, args)
    val_ds = None
    if val_raw is not None:
        val_ds, _ = prepare_probe_dataset(val_raw, tokenizer, args)

    quant_config = None
    if args.use_4bit or args.use_8bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=args.use_4bit,
            load_in_8bit=args.use_8bit,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        quantization_config=quant_config,
        attn_implementation="flash_attention_2" if args.use_flash_attn else None,
    )
    # Freeze base model
    for p in base_model.parameters():
        p.requires_grad = False

    hidden_size = int(base_model.config.hidden_size)
    head = ProbeHead(hidden_size=hidden_size, head_hidden_dim=args.head_hidden_dim, dropout=args.dropout, target_type=args.target_type)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model.to(device)
    head.to(device)

    # DataLoaders
    collator = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(train_ds, batch_size=args.per_device_train_batch_size, shuffle=True, collate_fn=collator)
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(val_ds, batch_size=args.per_device_eval_batch_size, shuffle=False, collate_fn=collator)

    # Optimizer & loss
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.target_type == "confidence":
        criterion = nn.MSELoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    # WandB
    use_wandb = bool(args.wandb_project)
    if use_wandb:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_run_name, dir=str(log_dir))

    global_step = 0
    best_metric = None
    for epoch in range(args.num_train_epochs):
        head.train()
        running_loss = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch[label_field].to(device).float()

            outputs = base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
            hs = outputs.hidden_states[-1]
            h_last = get_last_hidden(hs, attention_mask, args.pooling)
            logits = head(h_last)

            if args.target_type == "confidence":
                loss = criterion(torch.sigmoid(logits), labels)
            else:
                loss = criterion(logits, labels)

            loss = loss / max(1, args.gradient_accumulation_steps)
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            running_loss += float(loss.item())

            if use_wandb and global_step % args.logging_steps == 0:
                wandb.log({"train/loss": running_loss / max(1, args.logging_steps), "step": global_step, "epoch": epoch})
                running_loss = 0.0

        # Evaluate
        if val_loader is not None:
            metrics = evaluate(base_model, head, val_loader, args.pooling, args.target_type)
            if use_wandb:
                wandb.log({f"eval/{k}": v for k, v in metrics.items()})
            # Track best by AUROC (classification) or MSE (confidence)
            current = metrics.get("auroc", None) if args.target_type == "correctness" else metrics.get("mse", None)
            if current is not None:
                if best_metric is None or (
                    (args.target_type == "correctness" and current > best_metric) or
                    (args.target_type == "confidence" and current < best_metric)
                ):
                    best_metric = current
                    # Save best head
                    torch.save(head.state_dict(), output_dir / "probe_head.pt")

    # Save final head and config
    torch.save(head.state_dict(), output_dir / "probe_head_last.pt")
    meta = {
        "model_name_or_path": args.model_name_or_path,
        "target_type": args.target_type,
        "pooling": args.pooling,
        "head_hidden_dim": args.head_hidden_dim,
        "dropout": args.dropout,
        "hidden_size": hidden_size,
    }
    with open(output_dir / "probe_head_config.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
