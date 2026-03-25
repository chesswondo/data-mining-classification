import numpy as np
from typing import Dict, Any
from .base import BaseClassifier

class KNN(BaseClassifier):
    def __init__(self, k: int = 5, **kwargs: Dict[str, Any]):
        super().__init__(**kwargs)
        self.k = k
        self.X_train: np.ndarray = None
        self.y_train: np.ndarray = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KNN':
        self.X_train = X.copy()
        self.y_train = y.copy()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Model isn't trained. First call fit().")
            
        preds = []
        for x_test in X:
            # Calculate Euclidean distances
            distances = np.linalg.norm(self.X_train - x_test, axis=1)
            
            # Find k nearest neighbors
            k_indices = np.argpartition(distances, self.k)[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            
            # Election
            values, counts = np.unique(k_nearest_labels, return_counts=True)
            most_frequent = values[np.argmax(counts)]
            preds.append(most_frequent)
            
        return np.array(preds)