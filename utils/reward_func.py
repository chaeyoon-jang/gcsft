import math
import re
from typing import Iterable, List


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





def format_reward(format_pattern: str, completions: Iterable, **kwargs) -> List[float]:
    """
    Check if model-generated completion has correct format.
    Since think/answer are already in the prompt, we only check the generated part (confidence).
    Confidence is expected to be in range [0, 100].
    """
    confidence_pattern = r"(.*?)</confidence>"
    
    normalized = _normalize_completions(completions)
    completion_contents = [completion[0]["content"] for completion in normalized]
    matches = []
    
    for content in completion_contents:
        # Check if confidence tag exists and is properly formatted
        if "c" in format_pattern:
            confidence_matches = re.findall(confidence_pattern, content, re.DOTALL | re.MULTILINE)
            if not confidence_matches:
                matches.append(0.0)
            else:
                last_confidence = confidence_matches[-1]
                try:
                    confidence = float(last_confidence.strip())
                    # Check if confidence is in valid range [0, 100]
                    if 0 <= confidence <= 100:
                        matches.append(1.0)
                    else:
                        matches.append(0.0)
                except ValueError:
                    matches.append(0.0)
        else:
            # No confidence required in format pattern
            matches.append(1.0)
    return matches




def brier_reward(
    format_pattern: str,
    completions: Iterable,
    correctness: Iterable[int],
    **kwargs,
) -> List[float]:
    """
    Calculate Brier score as reward.
    Brier score = (confidence - correctness)^2
    where correctness is 0 or 1 from the data.
    Confidence is expected to be in range [0, 100] and will be normalized to [0, 1].
    If confidence is not found, returns 0 reward (penalty for not generating confidence).
    """
    confidence_pattern = r"(.*?)</confidence>"
    normalized = _normalize_completions(completions)
    completion_contents = [completion[0]["content"] for completion in normalized]
    matches = []
    
    correctness_list = list(correctness)
    
    for content, correctness_val in zip(completion_contents, correctness_list):
        confidence_matches = re.findall(confidence_pattern, content, re.DOTALL | re.MULTILINE)
        last_confidence = confidence_matches[-1] if confidence_matches else None
        
        if last_confidence is None:
            # No confidence found - return penalty (0 is worst)
            matches.append(-2.0)
        else:
            try:
                conf = float(last_confidence.strip())
                # Normalize confidence from [0, 100] to [0, 1]
                conf = conf * 0.01
                conf = max(0.0, min(conf, 1.0))
                # Brier score: (confidence - correctness)^2
                # Negate because lower brier is better, but GRPO maximizes reward
                brier = (conf - correctness_val) ** 2
                matches.append(-brier)
            except ValueError:
                # Parsing failed - return 0 reward (penalty)
                matches.append(0.0)

    return matches


def log_loss_reward(
    format_pattern: str,
    completions: Iterable,
    correctness: Iterable[int],
    epsilon: float = 1e-4,
    **kwargs,
) -> List[float]:
    """
    Calculate log-loss (negative log-likelihood) as reward.
    - If correctness = 1: reward = -log(confidence)
    - If correctness = 0: reward = -log(1 - confidence)
    Confidence is expected to be in range [0, 100] and will be normalized to [0, 1].
    If confidence is not found, returns 0 reward (penalty for not generating confidence).
    """
    confidence_pattern = r"<confidence>(.*?)</confidence>"
    normalized = _normalize_completions(completions)
    completion_contents = [completion[0]["content"] for completion in normalized]
    matches = []
    
    correctness_list = list(correctness)

    for content, correctness_val in zip(completion_contents, correctness_list):
        confidence_matches = re.findall(confidence_pattern, content, re.DOTALL | re.MULTILINE)
        last_confidence = confidence_matches[-1] if confidence_matches else None
        
        if last_confidence is None:
            # No confidence found - return penalty (0 is worst)
            matches.append(0.0)
        else:
            try:
                conf = float(last_confidence.strip())
                # Normalize confidence from [0, 100] to [0, 1]
                conf = conf * 0.01
                conf = max(0.0, min(conf, 1.0))
                # Log-loss calculation
                # Lower loss is better, but GRPO maximizes reward, so negate
                if correctness_val == 1:
                    # loss = log(1/confidence), negated for reward
                    loss = math.log(max(conf, epsilon))
                else:
                    # loss = log(1/(1-confidence)), negated for reward
                    loss = math.log(max(1 - conf, epsilon))
                matches.append(-loss)
            except ValueError:
                # Parsing failed - return 0 reward (penalty)
                matches.append(0.0)
    
    return matches
