import numpy as np
from typing import Dict, Any
from .base import BaseClassifier

class GaussianNaiveBayes(BaseClassifier):
    def __init__(self, var_smoothing: float = 1e-9, **kwargs: Dict[str, Any]):
        super().__init__(**kwargs)
        self.var_smoothing = float(var_smoothing)
        self.classes: np.ndarray = None
        self.mean: np.ndarray = None
        self.var: np.ndarray = None
        self.priors: np.ndarray = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GaussianNaiveBayes':
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]
        
        self.mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self.var = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors = np.zeros(n_classes, dtype=np.float64)
        
        # Calculate statistics for each class
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0) + self.var_smoothing
            
            # Prior probabilities: N_c / N_total
            self.priors[idx] = X_c.shape[0] / float(X.shape[0])
            
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.classes is None:
            raise RuntimeError("Model isn't trained. First call fit().")
            
        # Inference
        y_pred = [self._predict_single(x) for x in X]
        return np.array(y_pred)

    def _predict_single(self, x: np.ndarray) -> Any:
        posteriors = []
        
        # Calculate posterior probabilities
        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + posterior
            posteriors.append(posterior)
            
        # Return class with highest posterior
        return self.classes[np.argmax(posteriors)]

    def _pdf(self, class_idx: int, x: np.ndarray) -> np.ndarray:
        """
        Function to calculate probability density function.
        """
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        
        # Gaussian: (1 / sqrt(2 * pi * sigma^2)) * exp(-(x - mu)^2 / (2 * sigma^2))
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        
        pdf = numerator / denominator
        return np.clip(pdf, 1e-300, None)