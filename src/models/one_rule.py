import numpy as np
from typing import Dict, Any
from collections import Counter
from .base import BaseClassifier

class OneRule(BaseClassifier):
    def __init__(self, **kwargs: Dict[str, Any]):
        super().__init__(**kwargs)
        self.best_feature_idx: int = -1
        self.rules: dict = {}
        self.default_class: Any = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'OneRule':
        n_samples, n_features = X.shape
        
        # Determine the majority class
        counts = Counter(y)
        self.default_class = counts.most_common(1)[0][0]
        
        best_error = float('inf')
        
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            
            current_rules = {}
            errors = 0
            
            for val in unique_values:
                mask = (feature_values == val)
                target_subset = y[mask]
                
                most_frequent_class = Counter(target_subset).most_common(1)[0][0]
                current_rules[val] = most_frequent_class
                
                errors += np.sum(target_subset != most_frequent_class)
            
            if errors < best_error:
                best_error = errors
                self.best_feature_idx = feature_idx
                self.rules = current_rules
                
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.best_feature_idx == -1:
            raise RuntimeError("Model isn't trained. First call fit().")
            
        feature_col = X[:, self.best_feature_idx]
        preds = [self.rules.get(val, self.default_class) for val in feature_col]
        return np.array(preds)