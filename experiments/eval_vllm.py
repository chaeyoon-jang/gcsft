#!/usr/bin/env python
import argparse
import json
import time
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm.auto import tqdm

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from utils.prompt_hub import (get_answer_only_prompt, 
                        get_reasoning_prompt, 
                        get_confidence_prompt)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate models with vLLM")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--eval_file", default="openai/gsm8k")
    parser.add_argument("--split", default="main")
    parser.add_argument("--data_type", default="train")
    parser.add_argument("--instruction_type", default="reasoning")
    parser.add_argument("--output_file")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--lora_path")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--system_prompt", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log_dir", default=None)
    parser.add_argument("--add_conf", action="store_true")
    parser.add_argument("--confidence_prompt_name", default="default")
    parser.add_argument("--include_prompt", action="store_true",
                        help="Include original/processed prompt in output records")
    parser.add_argument("--dry_run", action="store_true",
                        help="Skip model inference and produce dummy outputs for validation")
    return parser.parse_args()


def load_jsonl(path: str) -> Dataset:
    with open(path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    return Dataset.from_list(records)


def load_any_dataset(path: str, split: str, data_type: str) -> Dataset:
    if path.endswith(".jsonl"):
        return load_jsonl(path)
    if path.endswith(".json"):
        return load_dataset("json", data_files=path, split="train")
    if path.endswith(".csv") or path.endswith(".tsv"):
        return Dataset.from_pandas(pd.read_csv(path))
    return load_dataset(path, split)[data_type]
'''
    if data_name == "gsm":
        data = datasets.load_dataset('openai/gsm8k', 'main')[data_type]
'''


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
    instruction_prompt: Optional[str],
    system_prompt: Optional[str],
) -> Tuple[str, Optional[List[Dict[str, str]]]]:
    if "prompt" in example:
        return example["prompt"], None
    if "question" in example:
        messages = [{"role": "user", "content": instruction_prompt\
                        + example["question"] if instruction_prompt 
                        else example["question"]}] 
        if system_prompt: 
            messages = [{"role": "system", "content": system_prompt}] + messages
        if tokenizer and use_chat_template:
            return (
                tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True,
                eable_thinking=True if "reasoning" in instruction_prompt else False
                ),
                messages,
            )
        return "\n".join([f"{m['role']}: {m['content']}" for m in messages]), messages
    raise ValueError("Example must contain 'prompt' or 'question'.")


def build_confidence_input(prompt: str, answer: str, prompt_name: str) -> str:
    conf_prompt = get_confidence_prompt(prompt_name)
    return f"{conf_prompt}\nQuestion:\n{prompt}\nAnswer:\n{answer}"


def extract_reference(example: Dict[str, Any]) -> Optional[str]:
    if "reference" in example:
        return example["reference"]
    if "response" in example:
        return example["response"]
    if "answer" in example:
        return example["answer"]
    if "outputs" in example:
        return example["outputs"]
    if "true_answer" in example:
        return example["true_answer"]
    return None


def main() -> None:
    args = parse_args()
    # Only resolve a log directory when no explicit output path is provided
    log_dir: Optional[Path] = None
    if not args.output_file:
        log_dir = resolve_log_dir(args.log_dir)

    overall_start = time.time()
    dataset = load_any_dataset(args.eval_file, split=args.split, data_type=args.data_type) # dataset file must include 'question' field
    needs_chat = args.use_chat_template or any("messages" in row for row in dataset)
    tokenizer = None
    if needs_chat:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
        
    instruction_prompt = get_reasoning_prompt("default") if args.instruction_type == "reasoning" else \
                         get_answer_only_prompt("default")
    prompt_entries = [
        build_prompt(row, tokenizer, args.use_chat_template, instruction_prompt, args.system_prompt)
        for row in dataset
    ]
    prompts = [entry[0] + "<think>" if args.instruction_type == "reasoning" else entry[0]\
        + "<answer>" for entry in prompt_entries]
    references = [extract_reference(row) for row in dataset]

    answer_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        stop=["</answer>"],
        seed=args.seed,
    )

    llm = None
    if not args.dry_run:
        llm = LLM(
            model=args.model_name_or_path,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_lora=bool(args.lora_path),
        )

    lora_request = None
    if args.lora_path:
        lora_request = LoRARequest("eval_adapter", 1, args.lora_path)

    outputs = []
    if args.dry_run:
        # Simulate progress without inference
        pbar = tqdm(total=len(prompts), desc="vLLM generation (dry-run)", unit="ex")
        if args.batch_size and args.batch_size > 0:
            for idx in range(0, len(prompts), args.batch_size):
                batch_prompts = prompts[idx : idx + args.batch_size]
                # Simulate latency
                time.sleep(0.01)
                outputs.extend([{"outputs": [{"text": ""}]} for _ in batch_prompts])
                pbar.update(len(batch_prompts))
        else:
            time.sleep(0.01)
            outputs = [{"outputs": [{"text": ""}]} for _ in prompts]
            pbar.update(len(prompts))
        pbar.close()
    else:
        if args.batch_size and args.batch_size > 0:
            pbar = tqdm(total=len(prompts), desc="vLLM generation", unit="ex")
            for idx in range(0, len(prompts), args.batch_size):
                batch_prompts = prompts[idx : idx + args.batch_size]
                outputs.extend(
                    llm.generate(batch_prompts, answer_params, lora_request=lora_request, use_tqdm=False)
                )
                pbar.update(len(batch_prompts))
            pbar.close()
        else:
            pbar = tqdm(total=len(prompts), desc="vLLM generation", unit="ex")
            outputs = llm.generate(prompts, answer_params, lora_request=lora_request, use_tqdm=False)
            pbar.update(len(prompts))
            pbar.close()

    records: List[Dict[str, Any]] = []
    exact_matches = 0
    confidence_outputs: List[Optional[str]] = [None] * len(prompts)
    if args.add_conf:
        confidence_prompts = [
            build_confidence_input(prompt, output.outputs[0].text, args.confidence_prompt_name)
            for prompt, output in zip(prompts, outputs)
        ]
        confidence_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
            stop=["</confidence>"],
            seed=args.seed,
        )
        if args.dry_run:
            # Simulate confidence generation
            pbar_conf = tqdm(total=len(confidence_prompts), desc="Confidence gen (dry-run)", unit="ex")
            if args.batch_size and args.batch_size > 0:
                conf_outputs = []
                for idx in range(0, len(confidence_prompts), args.batch_size):
                    conf_batch = confidence_prompts[idx : idx + args.batch_size]
                    time.sleep(0.01)
                    conf_outputs.extend([{"outputs": [{"text": ""}]} for _ in conf_batch])
                    pbar_conf.update(len(conf_batch))
            else:
                time.sleep(0.01)
                conf_outputs = [{"outputs": [{"text": ""}]} for _ in confidence_prompts]
                pbar_conf.update(len(confidence_prompts))
            pbar_conf.close()
            confidence_outputs = [out["outputs"][0]["text"] for out in conf_outputs]
        else:
            if args.batch_size and args.batch_size > 0:
                pbar_conf = tqdm(total=len(confidence_prompts), desc="Confidence generation", unit="ex")
                conf_outputs = []
                for idx in range(0, len(confidence_prompts), args.batch_size):
                    conf_batch = confidence_prompts[idx : idx + args.batch_size]
                    conf_outputs.extend(
                        llm.generate(conf_batch, confidence_params, lora_request=lora_request, use_tqdm=False)
                    )
                    pbar_conf.update(len(conf_batch))
                pbar_conf.close()
            else:
                pbar_conf = tqdm(total=len(confidence_prompts), desc="Confidence generation", unit="ex")
                conf_outputs = llm.generate(
                    confidence_prompts, confidence_params, lora_request=lora_request, use_tqdm=False
                )
                pbar_conf.update(len(confidence_prompts))
                pbar_conf.close()
            confidence_outputs = [out.outputs[0].text for out in conf_outputs]

    generated_texts: List[str]
    # Normalize outputs for dry-run vs real outputs
    if args.dry_run:
        generated_texts = [out["outputs"][0]["text"] for out in outputs]
    else:
        generated_texts = [out.outputs[0].text for out in outputs]

    for idx, (prompt, completion, reference) in enumerate(zip(prompts, generated_texts, references)):
        rec: Dict[str, Any] = {
            "input": prompt,  
            "completion": completion,
            "reference": reference,
            "confidence": confidence_outputs[idx],
        }
        if args.include_prompt:
            rec["prompt"] = prompt
        records.append(rec)
        #if reference:
        #    exact_matches += int(reference.strip() == completion.strip())

    overall_end = time.time()
    elapsed_seconds = overall_end - overall_start

    results = {
        "count": len(records),
        #"exact_match": exact_matches / len(records) if records else 0.0,
        "average_completion_length": sum(len(r["completion"]) for r in records) / len(records)
        if records
        else 0.0,
        "elapsed_seconds": elapsed_seconds,
    }

    output_path: Path
    if args.output_file:
        # Honor provided output path as-is; create parent directories as needed
        output_path = Path(args.output_file)
    else:
        # Fall back to a timestamped logs directory
        assert log_dir is not None
        output_path = log_dir / "eval_outputs.jsonl"

    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    with open(str(output_path) + ".metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Print concise timing summary for terminal visibility
    print(f"Processed {len(records)} examples in {elapsed_seconds:.2f}s. Outputs: {output_path}")


if __name__ == "__main__":
    main()