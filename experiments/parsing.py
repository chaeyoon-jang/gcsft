#!/usr/bin/env python
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from vllm import LLM, SamplingParams

DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"

SYSTEM_PROMPT = """You are a strict answer parser.
Return ONLY valid JSON with double quotes. Do not include markdown.
If information is missing, use null. Do not add extra keys.
"""

TASK_PROMPTS = {
    "ruler": """Extract the shortest correct answer span for a RULER-style QA task.
- Prefer exact spans that appear in the context or gold answer.
- If the prediction is verbose, keep only the final answer string.
- Preserve original casing for proper nouns.
Return JSON with keys: parsed_gold, parsed_pred.
""",
    "gsm8k_math": """Extract the final numeric answer.
- Strip units and words.
- If there is a fraction or decimal, keep it as a simplified string.
- If multiple numbers appear, choose the final answer.
Return JSON with keys: parsed_gold, parsed_pred.
""",
    "multiple_choice": """Extract the selected choice.
- Return a single letter (A/B/C/D/E/etc.).
- If the text includes the option content, map it to its letter.
- If ambiguous, return null.
Return JSON with keys: parsed_gold, parsed_pred.
""",
    "coding": """Extract the final answer for a coding task.
- If the gold/predicted answer includes code, keep only the code block contents.
- If it includes a final output/result, keep only that output.
- If both appear, prefer the final output.
Return JSON with keys: parsed_gold, parsed_pred.
""",
}

CHOICE_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H"]


@dataclass
class Record:
    question: str
    gold_answer: Optional[str]
    predicted_answer: Optional[str]
    task_type: str
    choices: Optional[List[str]] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse answers with vLLM.")
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL)
    parser.add_argument("--task_type", default=None,
                        choices=list(TASK_PROMPTS.keys()) + [None])
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_first(record: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for key in keys:
        if key in record and record[key] is not None:
            return str(record[key])
    return None


def normalize_task_type(task_type: Optional[str]) -> str:
    if task_type in TASK_PROMPTS:
        return task_type
    return "gsm8k_math"


def build_record(record: Dict[str, Any], default_task: Optional[str]) -> Record:
    task_type = normalize_task_type(record.get("task_type") or default_task)
    question = extract_first(record, ["question", "prompt", "input"]) or ""
    gold = extract_first(record, ["gold_answer", "gold", "answer", "reference", "label"])
    pred = extract_first(record, ["predicted_answer", "prediction", "pred", "response", "output"])
    choices = record.get("choices")
    return Record(question=question, gold_answer=gold, predicted_answer=pred,
                  task_type=task_type, choices=choices)


def format_choices(choices: Optional[List[str]]) -> str:
    if not choices:
        return ""
    lines = [f"{label}. {choice}" for label, choice in zip(CHOICE_LABELS, choices)]
    return "\nChoices:\n" + "\n".join(lines)


def build_user_prompt(record: Record) -> str:
    task_prompt = TASK_PROMPTS[record.task_type]
    choices_block = format_choices(record.choices)
    return (
        f"Task Type: {record.task_type}\n"
        f"Question:\n{record.question}{choices_block}\n\n"
        f"Golden Answer:\n{record.gold_answer}\n\n"
        f"Predicted Answer:\n{record.predicted_answer}\n\n"
        f"Instructions:\n{task_prompt}"
    )


def extract_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


def run_parsing(records: List[Record], args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.dry_run:
        return [
            {
                "task_type": rec.task_type,
                "question": rec.question,
                "gold_answer": rec.gold_answer,
                "predicted_answer": rec.predicted_answer,
                "parsed_gold": rec.gold_answer,
                "parsed_pred": rec.predicted_answer,
            }
            for rec in records
        ]

    llm = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    prompts = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(rec)},
        ]
        for rec in records
    ]

    outputs = []
    for idx in range(0, len(prompts), args.batch_size):
        batch = prompts[idx: idx + args.batch_size]
        outputs.extend(llm.chat(batch, sampling_params, use_tqdm=False))

    parsed_rows: List[Dict[str, Any]] = []
    for rec, output in zip(records, outputs):
        response_text = output.outputs[0].text
        parsed = extract_json(response_text)
        parsed_rows.append(
            {
                "task_type": rec.task_type,
                "question": rec.question,
                "gold_answer": rec.gold_answer,
                "predicted_answer": rec.predicted_answer,
                "parsed_gold": parsed.get("parsed_gold"),
                "parsed_pred": parsed.get("parsed_pred"),
            }
        )
    return parsed_rows


def main() -> None:
    args = parse_args()
    raw_records = read_jsonl(args.input_file)
    records = [build_record(row, args.task_type) for row in raw_records]
    parsed_rows = run_parsing(records, args)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(str(output_path), parsed_rows)


if __name__ == "__main__":
    main()
