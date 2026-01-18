from typing import Dict

REASONING_PROMPTS: Dict[str, str] = {
    "default": (
        "You are a reasoning assistant.\n"
        "Given a question, think step by step to arrive at the correct answer.\n"
        "First, provide your reasoning enclosed in <think> and </think> tags.\n"
        "Then, provide the final answer enclosed in <answer> and </answer> tags.\n"
        "The <answer> tag must contain only the final answer, with no additional explanation.\n\n"
    ),
}

ANSWER_ONLY_PROMPTS: Dict[str, str] = {
    "default": (
        "You are an answering assistant.\n"
        "Given a question, provide the final answer.\n"
        "Respond with the answer enclosed in <answer> and </answer> tags.\n"
        "The <answer> tag must contain only the final answer, with no additional explanation.\n\n"
    ),
}

CONFIDENCE_PROMPTS: Dict[str, str] = {
    "default": (
        "You are a confidence estimator.\n"
        "Given a question and a model answer, how confident are you that the answer is correct?\n"
        "Respond with a single integer from 0 to 100 enclosed in <confidence> and </confidence> tags.\n"
        "<confidence>"
    ),
    "multi": (
        "You are a confidence estimator.\n"
        "Given a question and a model answer, estimate confidence along two independent aspects.\n"
        "1. Reasoning Confidence:\n"
        "How confident are you that the reasoning process leading to the answer is logically sound and free of internal errors?\n"
        "2. Evidence Confidence:\n"
        "How confident are you that the answer is directly supported by the given context or evidence, if any?\n"
        "Respond with two integers from 0 to 100, each enclosed in the corresponding XML tags.\n"
        "Do not include any explanation or additional text.\n"
        "<reasoning_confidence>"
    ),
}

CHECK_PROMPTS: Dict[str, str] = {
    "evidence": (
        "\nI have made a best effort to use the available evidence, which may still be useful as a reference. However, I am not fully confident that all relevant information has been accounted for. User review is recommended."
    ),
    "reasoning": (
        "\nI have worked through the problem with a detailed solution process that may be helpful as a reference. However, I am not fully confident in the final answer, and the user is encouraged to review the solution carefully."
    ),
}

def get_confidence_prompt(name: str = "default") -> str:
    if name not in CONFIDENCE_PROMPTS:
        raise KeyError(f"Unknown confidence prompt: {name}")
    return CONFIDENCE_PROMPTS[name]

def get_check_prompt(name: str = "evidence") -> str:
    if name not in CHECK_PROMPTS:
        raise KeyError(f"Unknown check prompt: {name}")
    return CHECK_PROMPTS[name]

def get_reasoning_prompt(name: str = "default") -> str:
    if name not in REASONING_PROMPTS:
        raise KeyError(f"Unknown reasoning prompt: {name}")
    return REASONING_PROMPTS[name]

def get_answer_only_prompt(name: str = "default") -> str:
    if name not in ANSWER_ONLY_PROMPTS:
        raise KeyError(f"Unknown answer-only prompt: {name}")
    return ANSWER_ONLY_PROMPTS[name]