#!/usr/bin/env python
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate models with vLLM")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--output_file")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--lora_path")
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--system_prompt", default=None)
    parser.add_argument("--log_dir", default=None)
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


def build_prompt(
    example: Dict[str, Any],
    tokenizer: Optional[AutoTokenizer],
    use_chat_template: bool,
    system_prompt: Optional[str],
) -> str:
    if "prompt" in example:
        return example["prompt"]
    if "text" in example:
        return example["text"]
    if "messages" in example:
        messages = list(example["messages"])
        if system_prompt and not any(m.get("role") == "system" for m in messages):
            messages = [{"role": "system", "content": system_prompt}] + messages
        if tokenizer and use_chat_template:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    raise ValueError("Example must contain 'prompt', 'text', or 'messages'.")


def extract_reference(example: Dict[str, Any]) -> Optional[str]:
    if "reference" in example:
        return example["reference"]
    if "response" in example:
        return example["response"]
    if "answer" in example:
        return example["answer"]
    return None


def main() -> None:
    args = parse_args()
    log_dir = resolve_log_dir(args.log_dir)

    dataset = load_any_dataset(args.eval_file)
    needs_chat = args.use_chat_template or any("messages" in row for row in dataset)
    tokenizer = None
    if needs_chat:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    prompts = [
        build_prompt(row, tokenizer, args.use_chat_template, args.system_prompt)
        for row in dataset
    ]
    references = [extract_reference(row) for row in dataset]

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )

    llm = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_lora=bool(args.lora_path),
    )

    lora_request = None
    if args.lora_path:
        lora_request = LoRARequest(\"eval_adapter\", 1, args.lora_path)

    outputs = []
    if args.batch_size and args.batch_size > 0:
        for idx in range(0, len(prompts), args.batch_size):
            batch_prompts = prompts[idx : idx + args.batch_size]
            outputs.extend(
                llm.generate(batch_prompts, sampling_params, lora_request=lora_request)
            )
    else:
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

    records: List[Dict[str, Any]] = []
    exact_matches = 0
    for prompt, output, reference in zip(prompts, outputs, references):
        completion = output.outputs[0].text
        records.append(
            {
                "prompt": prompt,
                "completion": completion,
                "reference": reference,
            }
        )
        if reference:
            exact_matches += int(reference.strip() == completion.strip())

    results = {
        "count": len(records),
        "exact_match": exact_matches / len(records) if records else 0.0,
        "average_completion_length": sum(len(r["completion"]) for r in records) / len(records)
        if records
        else 0.0,
    }

    output_path = None
    if args.output_file:
        output_path = Path(args.output_file)
        if not output_path.is_absolute():
            output_path = log_dir / output_path
    else:
        output_path = log_dir / "eval_outputs.jsonl"

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    with open(str(output_path) + ".metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
