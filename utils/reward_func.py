import math
import re
import string
from typing import Iterable, List, Optional

from math_verify import parse, verify


def _normalize_completions(completions: Iterable) -> List[List[dict]]:
    normalized = []
    for completion in completions:
        if isinstance(completion, str):
            normalized.append([{"content": completion}])
        elif isinstance(completion, list):
            normalized.append(completion)
        else:
            raise TypeError("Completion must be a string or list of dicts.")
    return normalized


def normalize_answer(s: str) -> str:
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


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def format_reward(format_pattern: str, completions: Iterable, **kwargs) -> List[float]:
    if format_pattern == "tbac":
        pattern = r".*?</think>\s*<analysis>.*?</analysis>\s*<answer>.*?</answer>\s*<confidence>.*?</confidence>\s*\Z"
    elif format_pattern == "ta":
        pattern = r".*?</think>\s*<answer>.*?</answer>\s*\Z"
    elif format_pattern == "tac":
        pattern = r".*?</think>\s*<answer>.*?</answer>\s*<confidence>.*?</confidence>\s*\Z"
    elif format_pattern == "tabc":
        pattern = r".*?</think>\s*<answer>.*?</answer>\s*<analysis>.*?</analysis>\s*<confidence>.*?</confidence>\s*\Z"
    else:
        raise ValueError(f"Unknown format pattern: {format_pattern}")
    confidence_pattern = r"<confidence>(.*?)</confidence>"

    normalized = _normalize_completions(completions)
    completion_contents = [completion[0]["content"] for completion in normalized]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    matches = [1.0 if match else 0.0 for match in matches]

    for i, match in enumerate(matches):
        if match and "c" in format_pattern:
            content = completion_contents[i]
            confidence_matches = re.findall(confidence_pattern, content, re.DOTALL | re.MULTILINE)
            last_confidence = confidence_matches[-1] if confidence_matches else ""
            if last_confidence == "":
                matches[i] = 0.0
            else:
                try:
                    confidence = float(last_confidence)
                    matches[i] = 1.0 if 0 <= confidence <= 1 else 0.0
                except ValueError:
                    matches[i] = 0.0
    return matches


def accuracy_reward(
    format_pattern: str,
    completions: Iterable,
    answer: Iterable[str],
    source: Optional[Iterable[str]] = None,
    **kwargs,
) -> List[float]:
    ans_pattern = r"<answer>(.*?)</answer>"
    normalized = _normalize_completions(completions)
    completion_contents = [completion[0]["content"] for completion in normalized]
    eval_contents = [e for e in answer]
    matches = []
    format_rewards = format_reward(format_pattern, completions)
    source_entries = list(source) if source is not None else [None] * len(eval_contents)

    for content, e, fr, src in zip(completion_contents, eval_contents, format_rewards, source_entries):
        if fr == 0:
            matches.append(0)
        else:
            ans_matches = re.findall(ans_pattern, content, re.DOTALL | re.MULTILINE)
            last_answer = ans_matches[-1] if ans_matches else ""
            if src is not None and src == "hotpot":
                label = exact_match_score(last_answer, e)
            else:
                attempt = parse(last_answer)
                label = verify(e, attempt)
            matches.append(float(label))
    return matches


def brier_reward(
    format_pattern: str,
    completions: Iterable,
    answer: Iterable[str],
    source: Optional[Iterable[str]] = None,
    **kwargs,
) -> List[float]:
    confidence_pattern = r"<confidence>(.*?)</confidence>"
    normalized = _normalize_completions(completions)
    completion_contents = [completion[0]["content"] for completion in normalized]
    matches = []
    correctness_rewards = accuracy_reward(format_pattern, completions, answer, source)
    format_rewards = format_reward(format_pattern, completions)
    for content, cr, fr in zip(completion_contents, correctness_rewards, format_rewards):
        if fr == 0:
            matches.append(0)
        else:
            confidence_matches = re.findall(confidence_pattern, content, re.DOTALL | re.MULTILINE)
            last_confidence = confidence_matches[-1] if confidence_matches else ""
            if last_confidence == "":
                matches.append(0)
            else:
                try:
                    conf = float(last_confidence)
                    reward = 1 - (cr - conf) ** 2
                    matches.append(reward)
                except ValueError:
                    matches.append(0)
    return matches


def mean_confidence_reward(completions: Iterable, answer: Iterable[str], **kwargs) -> List[float]:
    confidence_pattern = r"<confidence>(.*?)</confidence>"
    normalized = _normalize_completions(completions)
    completion_contents = [completion[0]["content"] for completion in normalized]
    eval_contents = [e for e in answer]
    matches = []

    for content, _ in zip(completion_contents, eval_contents):
        confidence_matches = re.findall(confidence_pattern, content, re.DOTALL | re.MULTILINE)
        last_confidence = confidence_matches[-1] if confidence_matches else ""
        if last_confidence == "":
            matches.append(0.0)
        else:
            try:
                confidence = float(last_confidence)
                confidence = max(0.0, min(confidence, 1.0))
            except ValueError:
                confidence = 0.0
            matches.append(confidence)
    return matches


def confidence_one_or_zero(completions: Iterable, answer: Iterable[str], **kwargs) -> List[float]:
    confidence_pattern = r"<confidence>(.*?)</confidence>"
    normalized = _normalize_completions(completions)
    completion_contents = [completion[0]["content"] for completion in normalized]
    eval_contents = [e for e in answer]
    matches = []

    for content, _ in zip(completion_contents, eval_contents):
        confidence_matches = re.findall(confidence_pattern, content, re.DOTALL | re.MULTILINE)
        last_confidence = confidence_matches[-1] if confidence_matches else ""
        if last_confidence == "":
            matches.append(0.0)
        else:
            try:
                confidence = float(last_confidence)
                confidence = max(0.0, min(confidence, 1.0))
            except ValueError:
                confidence = 0.0
            if abs(confidence - 1) < 0.01 or abs(confidence - 0) < 0.01:
                matches.append(1.0)
            else:
                matches.append(0.0)
    return matches


def rlcr_reward(
    reward_type: str,
    format_pattern: str,
    completions: Iterable,
    answer: Iterable[str],
    source: Optional[Iterable[str]] = None,
    **kwargs,
) -> List[float]:
    if reward_type == "format":
        return format_reward(format_pattern, completions)
    if reward_type == "accuracy":
        return accuracy_reward(format_pattern, completions, answer, source)
    if reward_type == "brier":
        return brier_reward(format_pattern, completions, answer, source)
    if reward_type == "mean_confidence":
        return mean_confidence_reward(completions, answer)
    if reward_type == "confidence_one_or_zero":
        return confidence_one_or_zero(completions, answer)
    raise ValueError(f"Unknown RLCR reward type: {reward_type}")


def rewarding_doubt_reward(
    format_pattern: str,
    completions: Iterable,
    answer: Iterable[str],
    source: Optional[Iterable[str]] = None,
    epsilon: float = 1e-4,
    **kwargs,
) -> List[float]:
    confidence_pattern = r"<confidence>(.*?)</confidence>"
    normalized = _normalize_completions(completions)
    completion_contents = [completion[0]["content"] for completion in normalized]
    correctness_rewards = accuracy_reward(format_pattern, completions, answer, source)
    format_rewards = format_reward(format_pattern, completions)
    matches = []

    for content, correctness, format_reward_value in zip(
        completion_contents, correctness_rewards, format_rewards
    ):
        if format_reward_value == 0:
            matches.append(0.0)
            continue
        confidence_matches = re.findall(confidence_pattern, content, re.DOTALL | re.MULTILINE)
        last_confidence = confidence_matches[-1] if confidence_matches else ""
        if last_confidence == "":
            matches.append(0.0)
            continue
        try:
            conf = float(last_confidence)
        except ValueError:
            matches.append(0.0)
            continue
        if correctness == 1:
            reward = math.log(max(conf, epsilon))
        else:
            reward = math.log(min(1 - conf, 1 - epsilon))
        matches.append(reward)
    return matches
