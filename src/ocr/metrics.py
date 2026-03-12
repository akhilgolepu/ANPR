"""
OCR Evaluation Metrics
Compute accuracy, character-level accuracy, confidence distributions, etc.
"""

import re
from typing import List, Dict, Any
from collections import defaultdict

import numpy as np


def normalize_plate_text(text: str) -> str:
    """Normalize plate text for comparison (remove spaces, uppercase)."""
    return re.sub(r'[^A-Z0-9]', '', text.upper())


def compute_char_accuracy(predicted: str, ground_truth: str) -> float:
    """Compute character-level accuracy using simple comparison."""
    if not ground_truth:
        return 1.0 if not predicted else 0.0
    
    pred_norm = normalize_plate_text(predicted)
    gt_norm = normalize_plate_text(ground_truth)
    
    if len(gt_norm) == 0:
        return 1.0 if len(pred_norm) == 0 else 0.0
    
    # Count matching characters (order matters)
    matches = sum(1 for p, g in zip(pred_norm, gt_norm) if p == g)
    # Use max length for denominator
    max_len = max(len(pred_norm), len(gt_norm))
    
    return matches / max_len if max_len > 0 else 0.0


def compute_ocr_metrics(predictions: List[str], ground_truths: List[str]) -> Dict[str, Any]:
    """
    Compute OCR evaluation metrics.
    
    Args:
        predictions: List of recognized plate texts
        ground_truths: List of ground truth plate texts
    
    Returns:
        Dictionary with accuracy, character accuracy, and detailed stats
    """
    if not predictions:
        return {"accuracy": 0.0, "char_accuracy": 0.0}
    
    # Exact match accuracy (entire plate)
    exact_matches = sum(1 for pred, gt in zip(predictions, ground_truths) 
                       if normalize_plate_text(pred) == normalize_plate_text(gt))
    accuracy = exact_matches / len(predictions)
    
    # Character-level accuracy
    char_accs = [compute_char_accuracy(p, gt) for p, gt in zip(predictions, ground_truths)]
    char_accuracy = np.mean(char_accs)
    
    # Plate length distribution
    gt_lengths = [len(normalize_plate_text(gt)) for gt in ground_truths]
    pred_lengths = [len(normalize_plate_text(p)) for p in predictions]
    
    return {
        "accuracy": float(accuracy),
        "char_accuracy": float(char_accuracy),
        "total_samples": len(predictions),
        "exact_matches": int(exact_matches),
        "avg_ground_truth_length": float(np.mean(gt_lengths)) if gt_lengths else 0.0,
        "avg_predicted_length": float(np.mean(pred_lengths)) if pred_lengths else 0.0,
    }
