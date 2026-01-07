import numpy as np
from sklearn.metrics import roc_auc_score
from typing import Union


def ece(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE).
    
    Measures the difference between predicted confidence and actual accuracy.
    
    Args:
        y_true: True binary labels (0 or 1)
        y_proba: Predicted probabilities for the positive class
        n_bins: Number of bins for calibration calculation
        
    Returns:
        ECE value between 0 and 1
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    ece_value = 0.0
    for i in range(n_bins):
        mask = (y_proba >= bin_edges[i]) & (y_proba < bin_edges[i + 1])
        if mask.sum() > 0:
            accuracy = y_true[mask].mean()
            confidence = y_proba[mask].mean()
            ece_value += mask.sum() / len(y_true) * np.abs(accuracy - confidence)
    
    return ece_value


def brier_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Brier Score.
    
    Mean squared error between predicted probabilities and actual outcomes.
    Lower values indicate better calibration.
    
    Args:
        y_true: True binary labels (0 or 1)
        y_proba: Predicted probabilities for the positive class
        
    Returns:
        Brier score between 0 and 1
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    
    return np.mean((y_proba - y_true) ** 2)


def nll(y_true: np.ndarray, y_proba: np.ndarray, epsilon: float = 1e-15) -> float:
    """
    Negative Log Likelihood (NLL).
    
    Cross-entropy loss for binary classification.
    
    Args:
        y_true: True binary labels (0 or 1)
        y_proba: Predicted probabilities for the positive class
        epsilon: Small value to avoid log(0)
        
    Returns:
        Negative log likelihood (average cross-entropy loss)
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    
    # Clip probabilities to avoid log(0)
    y_proba = np.clip(y_proba, epsilon, 1 - epsilon)
    
    # Binary cross-entropy
    nll_value = -np.mean(y_true * np.log(y_proba) + (1 - y_true) * np.log(1 - y_proba))
    
    return nll_value


def auroc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Area Under the Receiver Operating Characteristic Curve (AUROC).
    
    Measures the probability that the model ranks a random positive example
    higher than a random negative example.
    
    Args:
        y_true: True binary labels (0 or 1)
        y_proba: Predicted probabilities for the positive class
        
    Returns:
        AUROC score between 0 and 1
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    
    # Handle edge cases
    if len(np.unique(y_true)) < 2:
        return np.nan
    
    return roc_auc_score(y_true, y_proba)
