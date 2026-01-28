import math
import re
from typing import List, Optional


def extract_answer(completion: str) -> Optional[str]:
    """
    Extract answer from <answer>...</answer> tags.
    Returns the content inside the last occurrence of answer tags.
    """
    pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(pattern, completion, re.DOTALL | re.IGNORECASE)
    
    if not matches:
        return None
    
    # Return the last match (in case there are multiple)
    return matches[-1].strip()


def extract_confidence(completion: str) -> Optional[float]:
    """
    Extract confidence from <confidence>...</confidence> tags.
    Expects confidence in range [0, 100], returns normalized to [0, 1].
    """
    pattern = r"<confidence>\s*([0-9]+(?:\.[0-9]+)?)\s*</confidence>"
    matches = re.findall(pattern, completion, re.DOTALL | re.IGNORECASE)
    
    if not matches:
        return None
    
    try:
        # Use the last match
        conf_value = float(matches[-1])
        
        # Normalize from [0, 100] to [0, 1]
        conf_normalized = conf_value / 100.0
        
        # Clip to valid range
        return max(0.0, min(1.0, conf_normalized))
    except (ValueError, TypeError):
        return None


def check_correctness(predicted_answer: str, true_answer: str) -> bool:
    """
    Compare predicted answer with true answer.
    Simple exact match after stripping whitespace.
    """
    if predicted_answer is None or true_answer is None:
        return False
    
    return predicted_answer.strip().lower() in true_answer.strip().lower()


def brier_reward(
    completions: List[str],
    true_answers: List[str],
    missing_penalty: float = -10.0,
    reward_clip: tuple = (-1.0, 0.0),
    **kwargs,
) -> List[float]:
    """
    Calculate Brier score reward: -((confidence - correctness)^2)
    
    Args:
        completions: List of model completions
        true_answers: List of ground truth answers
        missing_penalty: Penalty when answer or confidence is missing
        reward_clip: (min, max) tuple to clip rewards
    
    Returns:
        List of reward values
    """
    rewards = []
    
    for completion, true_answer in zip(completions, true_answers):
        # Extract answer and confidence
        predicted_answer = extract_answer(completion)
        confidence = extract_confidence(completion)
        
        # Check if extraction failed
        if predicted_answer is None or confidence is None:
            rewards.append(missing_penalty)
            continue
        
        # Check correctness (1.0 if correct, 0.0 if wrong)
        is_correct = 1.0 if check_correctness(predicted_answer, true_answer) else 0.0
        
        # Calculate Brier score: negative squared error
        reward = -((confidence - is_correct) ** 2)
        
        # Clip reward to valid range
        reward = max(reward_clip[0], min(reward, reward_clip[1]))
        
        if is_correct:
            reward += 0.25
            
        rewards.append(reward)
    
    return rewards


def log_loss_reward(
    completions: List[str],
    true_answers: List[str],
    epsilon: float = 1e-4,
    missing_penalty: float = -10.0,
    reward_clip: tuple = (-6.0, 0.0),
    **kwargs,
) -> List[float]:
    """
    Calculate log loss reward:
      - If correct: log(confidence)
      - If wrong: log(1 - confidence)
    
    Args:
        completions: List of model completions
        true_answers: List of ground truth answers
        epsilon: Small value to avoid log(0)
        missing_penalty: Penalty when answer or confidence is missing
        reward_clip: (min, max) tuple to clip rewards
    
    Returns:
        List of reward values
    """
    rewards = []
    
    for completion, true_answer in zip(completions, true_answers):
        # Extract answer and confidence
        predicted_answer = extract_answer(completion)
        confidence = extract_confidence(completion)
        
        # Check if extraction failed
        if predicted_answer is None or confidence is None:
            rewards.append(missing_penalty)
            continue
        
        # Check correctness
        is_correct = check_correctness(predicted_answer, true_answer)
        
        # Calculate log loss
        if is_correct:
            # Correct: reward = log(confidence)
            reward = math.log(max(confidence, epsilon))
        else:
            # Wrong: reward = log(1 - confidence)
            reward = math.log(max(1.0 - confidence, epsilon))
        
        # Clip reward to valid range
        reward = max(reward_clip[0], min(reward, reward_clip[1]))
        
        if is_correct:
            reward += 0.25 
            
        rewards.append(reward)
    
    return rewards