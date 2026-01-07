from typing import Dict


CONFIDENCE_PROMPTS: Dict[str, str] = {
    "default": (
        "You are a confidence estimator. Given a question and a model answer, output a"
        " confidence score between 0 and 1 with a brief rationale. End your response"
        " with </confidence>."
    ),
}


def get_confidence_prompt(name: str = "default") -> str:
    if name not in CONFIDENCE_PROMPTS:
        raise KeyError(f"Unknown confidence prompt: {name}")
    return CONFIDENCE_PROMPTS[name]
