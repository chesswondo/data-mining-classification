import numpy as np
from typing import Tuple

def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(y_true == y_pred)

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    num_classes = max(np.max(y_true), np.max(y_pred)) + 1
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix

def precision_recall_f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    classes = np.unique(np.concatenate((y_true, y_pred)))
    precisions, recalls, f1s = [], [], []
    
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
        
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
        
    return np.mean(precisions), np.mean(recalls), np.mean(f1s)