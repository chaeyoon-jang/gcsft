from typing import Dict

REASONING_PROMPTS: Dict[str, str] = {
    "default": (
        "You are a reasoning assistant.\n"
        "Given a question, think step by step to arrive at the correct answer.\n"
        "First, provide your reasoning enclosed in <think> and </think> tags.\n"
        "Then, provide the final answer enclosed in <answer> and </answer> tags.\n"
        "The <answer> tag must contain only the final answer, with no additional explanation."
    ),
}

ANSWER_ONLY_PROMPTS: Dict[str, str] = {
    "default": (
        "You are an answering assistant.\n"
        "Given a question, provide the final answer.\n"
        "Respond with the answer enclosed in <answer> and </answer> tags.\n"
        "The <answer> tag must contain only the final answer, with no additional explanation."
    ),
}

CONFIDENCE_PROMPTS: Dict[str, str] = {
    "default": (
        "You are a confidence estimator.\n"
        "Given a question and a model answer, how confident are you that the answer is correct?\n"
        "Respond with a single integer from 0 to 100 enclosed in <confidence> and </confidence> tags.\n"
        "<confidence>"
    ),
}

def get_confidence_prompt(name: str = "default") -> str:
    if name not in CONFIDENCE_PROMPTS:
        raise KeyError(f"Unknown confidence prompt: {name}")
    return CONFIDENCE_PROMPTS[name]

def get_reasoning_prompt(name: str = "default") -> str:
    if name not in REASONING_PROMPTS:
        raise KeyError(f"Unknown reasoning prompt: {name}")
    return REASONING_PROMPTS[name]

def get_answer_only_prompt(name: str = "default") -> str:
    if name not in ANSWER_ONLY_PROMPTS:
        raise KeyError(f"Unknown answer-only prompt: {name}")
    return ANSWER_ONLY_PROMPTS[name]