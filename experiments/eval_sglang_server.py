#!/usr/bin/env python
import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from tqdm.auto import tqdm
from datasets import Dataset, load_dataset

from utils.prompt_hub import get_confidence_prompt

from sglang.utils import execute_shell_command, wait_for_server
import sys

# Suppress sglang verbose logging
logging.getLogger("sglang").setLevel(logging.WARNING)
logging.getLogger("sglang.srt").setLevel(logging.WARNING)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate models with sglang server")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--launch_server", action="store_true")
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--system_prompt", default=None)
    parser.add_argument("--add_conf", action="store_true")
    parser.add_argument("--confidence_prompt_name", default="default")
    parser.add_argument("--include_prompt", action="store_true",
                        help="Include original/processed prompt in output records")
    parser.add_argument("--dry_run", action="store_true",
                        help="Skip model requests and produce dummy outputs for validation")
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def load_jsonl(path: str) -> Dataset:
    with open(path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    return Dataset.from_list(records)


def load_any_dataset(path: str) -> Dataset:
    """Load dataset from various formats: jsonl, json, csv, tsv, or HuggingFace dataset."""
    # Check file exists first
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    
    if path.endswith(".jsonl"):
        return load_jsonl(path)
    elif path.endswith(".json"):
        return load_dataset("json", data_files=path, split="train")
    elif path.endswith(".csv"):
        df = pd.read_csv(path)
        return Dataset.from_pandas(df)
    elif path.endswith(".tsv"):
        df = pd.read_csv(path, sep="\t")
        return Dataset.from_pandas(df)
    else:
        # Try to load as HuggingFace dataset
        return load_dataset(path, split="train")


def resolve_output_path(output_file: Optional[str], output_dir: Optional[str], seed: int) -> Path:
    """Resolve output file path under ./logs directory or custom output_dir."""
    if output_dir:
        base_dir = Path(output_dir)
    else:
        base_dir = Path("logs")
    
    base_dir.mkdir(parents=True, exist_ok=True)
    
    if output_file:
        output_path = Path(output_file)
        if output_path.is_absolute():
            return output_path
        return base_dir / output_file
    else:
        return base_dir / f"outputs_seed_{seed}.jsonl"


def build_prompt(
    example: Dict[str, Any],
    system_prompt: Optional[str],
) -> Tuple[str, List[Dict[str, str]]]:
    """Build prompt from example, similar to eval_vllm.py approach."""
    if "prompt" in example:
        # Direct prompt field
        messages = [{"role": "user", "content": example["prompt"]}]
        return example["prompt"], messages
    elif "question" in example:
        # Question field (convert to messages if needed)
        question_text = example["question"]
        messages = [{"role": "user", "content": question_text}]
        if system_prompt and not any(m.get("role") == "system" for m in messages):
            messages = [{"role": "system", "content": system_prompt}] + messages
        prompt_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        return prompt_text, messages
    elif "messages" in example:
        # Already formatted as messages
        messages = list(example["messages"])
        if system_prompt and not any(m.get("role") == "system" for m in messages):
            messages = [{"role": "system", "content": system_prompt}] + messages
        prompt_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        return prompt_text, messages
    elif "text" in example:
        # Text field
        text = example["text"]
        messages = [{"role": "user", "content": text}]
        if system_prompt and not any(m.get("role") == "system" for m in messages):
            messages = [{"role": "system", "content": system_prompt}] + messages
        prompt_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        return prompt_text, messages
    else:
        raise ValueError("Example must contain 'prompt', 'question', 'messages', or 'text'.")


def extract_reference(example: Dict[str, Any]) -> Optional[str]:
    if "reference" in example:
        return example["reference"]
    if "response" in example:
        return example["response"]
    if "answer" in example:
        return example["answer"]
    return None


def build_confidence_input(prompt: str, answer: str, prompt_name: str) -> str:
    conf_prompt = get_confidence_prompt(prompt_name)
    return f"{conf_prompt}\nQuestion:\n{prompt}\nAnswer:\n{answer}"


def request_completion(
    base_url: str,
    messages: List[Dict[str, str]],
    temperature: float,
    top_p: float,
    max_tokens: int,
    stop: List[str],
) -> str:
    response = requests.post(
        f"{base_url}/chat/completions",
        json={
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stop": stop,
        },
        timeout=600,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def run_eval(args: argparse.Namespace, seed: int) -> Path:
    eval_start = time.time()
    dataset = load_any_dataset(args.eval_file)

    prompt_entries = [build_prompt(row, args.system_prompt) for row in dataset]
    prompt_texts = [entry[0] for entry in prompt_entries]
    prompt_messages = [entry[1] for entry in prompt_entries]
    references = [extract_reference(row) for row in dataset]

    answers = []
    if args.dry_run:
        for _ in tqdm(prompt_messages, desc=f"Answers (dry-run, seed={seed})"):
            time.sleep(0.001)
            answers.append("")
    else:
        for messages in tqdm(prompt_messages, desc=f"Answers (seed={seed})"):
            answers.append(
                request_completion(
                    args.base_url,
                    messages,
                    args.temperature,
                    args.top_p,
                    args.max_new_tokens,
                    ["</answer>"],
                )
            )

    confidences: List[Optional[str]] = [None] * len(answers)
    if args.add_conf:
        if args.dry_run:
            for idx in tqdm(range(len(answers)), desc=f"Confidences (dry-run, seed={seed})"):
                time.sleep(0.001)
                confidences[idx] = ""
        else:
            for idx, (prompt, answer) in enumerate(tqdm(zip(prompt_texts, answers), total=len(answers), desc=f"Confidences (seed={seed})")):
                conf_input = build_confidence_input(prompt, answer, args.confidence_prompt_name)
                conf_messages = [{"role": "user", "content": conf_input}]
                confidences[idx] = request_completion(
                    args.base_url,
                    conf_messages,
                    args.temperature,
                    args.top_p,
                    args.max_new_tokens,
                    ["</confidence>"],
                )

    records: List[Dict[str, Any]] = []
    exact_matches = 0
    for prompt, answer, reference, confidence in tqdm(
        zip(prompt_texts, answers, references, confidences),
        total=len(answers),
        desc=f"Saving (seed={seed})"
    ):
        rec: Dict[str, Any] = {
            "completion": answer,
            "reference": reference,
            "confidence": confidence,
        }
        if args.include_prompt:
            rec["prompt"] = prompt
        records.append(rec)
        if reference:
            exact_matches += int(reference.strip() == answer.strip())

    eval_end = time.time()
    elapsed_seconds = eval_end - eval_start

    results = {
        "count": len(records),
        "exact_match": exact_matches / len(records) if records else 0.0,
        "average_completion_length": sum(len(r["completion"]) for r in records) / len(records)
        if records
        else 0.0,
        "elapsed_seconds": elapsed_seconds,
    }

    output_path = resolve_output_path(args.output_file, args.output_dir, seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    with open(str(output_path) + ".metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Processed {len(records)} examples in {elapsed_seconds:.2f}s. Outputs: {output_path}")
    return output_path


def main() -> None:
    args = parse_args()
    
    if args.launch_server and not args.model_name_or_path:
        print("[ERROR] --model_name_or_path is required when using --launch_server")
        sys.exit(1)
    
    server_proc = None
    try:
        if args.launch_server:
            cmd = (
                f"python3 -m sglang.launch_server "
                f"--model-path {args.model_name_or_path} "
                f"--host {args.host} --port {args.port} --tp {args.tp} --random-seed {args.seed}"
            )
            print(f"[INFO] Launching model server:\n  {cmd}")
            server_proc = execute_shell_command(cmd)
            wait_for_server(f"http://{args.host}:{args.port}")
            
        args.base_url = f"http://{args.host}:{args.port}/v1"
        
        # Single-seed execution
        print(f"[INFO] Running evaluation for seed: {args.seed}")
        run_eval(args, args.seed)
            
    except Exception as e:
        print(f"[ERROR] {e}")
        raise
    finally:
        if server_proc is not None:
            print("[INFO] Terminating model server...")
            server_proc.terminate()

if __name__ == "__main__":
    main()
