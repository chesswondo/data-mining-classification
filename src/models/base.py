from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Dict

class BaseClassifier(ABC):
    def __init__(self, **kwargs: Dict[str, Any]):
        """
        Base model constructor.
        """
        self.params = kwargs

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseClassifier':
        """
        Base training method.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Base inference. Returns predicted classes.
        """
        pass