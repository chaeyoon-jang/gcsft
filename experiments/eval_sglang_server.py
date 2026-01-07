#!/usr/bin/env python
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from datasets import Dataset, load_dataset

from prompt_hub import get_confidence_prompt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate models with sglang server")
    parser.add_argument("--base_url", default="http://localhost:30000/v1")
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--output_file")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--system_prompt", default=None)
    parser.add_argument("--log_dir", default=None)
    parser.add_argument("--add_conf", action="store_true")
    parser.add_argument("--confidence_prompt_name", default="default")
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


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
    system_prompt: Optional[str],
) -> Tuple[str, List[Dict[str, str]]]:
    if "messages" in example:
        messages = list(example["messages"])
    elif "prompt" in example:
        messages = [{"role": "user", "content": example["prompt"]}]
    elif "text" in example:
        messages = [{"role": "user", "content": example["text"]}]
    else:
        raise ValueError("Example must contain 'prompt', 'text', or 'messages'.")

    if system_prompt and not any(m.get("role") == "system" for m in messages):
        messages = [{"role": "system", "content": system_prompt}] + messages

    prompt_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    return prompt_text, messages


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


def run_eval(args: argparse.Namespace) -> Path:
    log_dir = resolve_log_dir(args.log_dir)
    dataset = load_any_dataset(args.eval_file)

    prompt_entries = [build_prompt(row, args.system_prompt) for row in dataset]
    prompt_texts = [entry[0] for entry in prompt_entries]
    prompt_messages = [entry[1] for entry in prompt_entries]
    references = [extract_reference(row) for row in dataset]

    answers = []
    for messages in prompt_messages:
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
        for idx, (prompt, answer) in enumerate(zip(prompt_texts, answers)):
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
    for prompt, answer, reference, confidence in zip(
        prompt_texts, answers, references, confidences
    ):
        records.append(
            {
                "prompt": prompt,
                "completion": answer,
                "reference": reference,
                "confidence": confidence,
            }
        )
        if reference:
            exact_matches += int(reference.strip() == answer.strip())

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

    return output_path


def main() -> None:
    args = parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
