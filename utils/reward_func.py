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


def strict_confidence_reward(completions: Iterable, **kwargs) -> List[float]:
    """
    Reward only if the completion ends with a well-formed confidence tag.
    Expected suffix: <confidence>0-100</confidence> with no trailing text.
    """
    pattern = r"<confidence>\s*([0-9]+(?:\.[0-9]+)?)\s*</confidence>\s*$"

    normalized = _normalize_completions(completions)
    completion_contents = [completion[0]["content"] for completion in normalized]
    rewards: List[float] = []

    for content in completion_contents:
        match = re.search(pattern, content, re.DOTALL | re.MULTILINE)
        if not match:
            rewards.append(0.0)
            continue

        try:
            conf_val = float(match.group(1))
        except ValueError:
            rewards.append(0.0)
            continue

        if 0.0 <= conf_val <= 100.0:
            rewards.append(1.0)
        else:
            rewards.append(0.0)

    return rewards




import re, math
from typing import Iterable, List, Optional, Tuple

# completion starts right after "<confidence>" (opening tag is in the prompt)
# So the first thing the model outputs should be a number, then </confidence>.
_START_CLOSE_RE = re.compile(
    r"^\s*([+-]?\d+(?:\.\d+)?)\s*%?\s*</confidence>",
    flags=re.IGNORECASE | re.DOTALL,
)

# optional fallback: model outputs its own <confidence> ... </confidence>
_TAGGED_RE = re.compile(
    r"<confidence>\s*([+-]?\d+(?:\.\d+)?)\s*%?\s*</confidence>",
    flags=re.IGNORECASE | re.DOTALL,
)

def _extract_confidence_0_1_from_completion_start(content: str) -> Optional[float]:
    """
    Enforce: the FIRST token after the provided '<confidence>' (i.e., at the start of completion)
    must be numeric and followed by '</confidence>'.
    Confidence is assumed in [0, 100], mapped to [0, 1].
    """
    if not content:
        return None

    m = _START_CLOSE_RE.search(content)
    if m:
        raw = m.group(1)
    else:
        # fallback (optional): if the model redundantly prints <confidence>...</confidence> itself
        ms = _TAGGED_RE.findall(content)
        if not ms:
            return None
        raw = ms[-1]

    try:
        conf_100 = float(raw)
    except ValueError:
        return None

    conf = conf_100 * 0.01
    if math.isnan(conf) or math.isinf(conf):
        return None
    return max(0.0, min(conf, 1.0))


def brier_reward(
    completions: Iterable,
    correctness: Iterable[int],
    missing_penalty: float = -1.25,
    invalid_penalty: float = -1.25,
    reward_clip: Tuple[float, float] = (-1.0, 0.0),
    **kwargs,
) -> List[float]:
    """
    Reward = -(conf - y)^2, where conf in [0,1], y in {0,1}.
    Enforces that completion starts with numeric confidence followed by </confidence>.
    Missing/invalid gets strong negative penalty (prevents 'skip' hacking).
    """
    normalized = _normalize_completions(completions)
    contents = [c[0]["content"] for c in normalized]
    ys = list(correctness)

    out: List[float] = []
    for content, y in zip(contents, ys):
        conf = _extract_confidence_0_1_from_completion_start(content)

        if conf is None:
            # differentiate "no closing tag at all" vs "has closing but malformed"
            pen = missing_penalty if "</confidence>" not in (content or "") else invalid_penalty
            r = pen
        else:
            r = -((conf - float(y)) ** 2)  # in [-1, 0]

        lo, hi = reward_clip
        r = max(lo, min(r, hi))
        out.append(float(r))

    return out


def log_loss_reward(
    completions: Iterable,
    correctness: Iterable[int],
    epsilon: float = 1e-4,
    missing_penalty: float = -10.0,
    invalid_penalty: float = -10.0,
    reward_clip: Tuple[float, float] = (-10.0, 0.0),
    **kwargs,
) -> List[float]:
    """
    Reward keeps original log-loss semantics but in reward form:
      y=1: reward = log(conf)
      y=0: reward = log(1-conf)
    Both <= 0. Uses epsilon + clipping for stability and anti-exploit.
    Enforces completion starts with numeric confidence followed by </confidence>.
    """
    normalized = _normalize_completions(completions)
    contents = [c[0]["content"] for c in normalized]
    ys = list(correctness)

    out: List[float] = []
    for content, y in zip(contents, ys):
        conf = _extract_confidence_0_1_from_completion_start(content)

        if conf is None:
            pen = missing_penalty if "</confidence>" not in (content or "") else invalid_penalty
            r = pen
        else:
            if int(y) == 1:
                r = math.log(max(conf, epsilon))          # <= 0
            else:
                r = math.log(max(1.0 - conf, epsilon))    # <= 0

        lo, hi = reward_clip
        r = max(lo, min(r, hi))
        out.append(float(r))

    return out