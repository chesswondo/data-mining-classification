import numpy as np
from typing import Dict, Any, Optional, Tuple
from collections import Counter
from .base import BaseClassifier

class Node:
    """
    Data structure, which represents a node in a decision tree.
    """
    def __init__(self, feature: Optional[int] = None, threshold: Optional[float] = None, 
                 left: Optional['Node'] = None, right: Optional['Node'] = None, *, 
                 value: Optional[Any] = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self) -> bool:
        return self.value is not None


class DecisionTree(BaseClassifier):
    def __init__(self, min_samples_split: int = 2, max_depth: int = 100, **kwargs: Dict[str, Any]):
        super().__init__(**kwargs)
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root: Optional[Node] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTree':
        self.root = self._build_tree(X, y, depth=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise RuntimeError("Model isn't trained. First call fit().")

        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Check stopping criteria
        if (depth >= self.max_depth or 
            n_labels == 1 or 
            n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Greedy search for the best split
        best_feat, best_thresh = self._best_split(X, y, n_features)

        # If split doesn't help (IG = 0), return leaf
        if best_feat is None:
            return Node(value=self._most_common_label(y))

        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._build_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._build_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        
        return Node(best_feat, best_thresh, left, right)

    def _best_split(self, X: np.ndarray, y: np.ndarray, n_features: int) -> Tuple[Optional[int], Optional[float]]:
        best_gain = -1
        split_idx, split_thresh = None, None
        
        for feat_idx in range(n_features):
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold
                    
        return split_idx, split_thresh

    def _information_gain(self, y: np.ndarray, X_column: np.ndarray, threshold: float) -> float:
        parent_entropy = self._entropy(y)
        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # Information gain
        return parent_entropy - child_entropy

    def _split(self, X_column: np.ndarray, split_thresh: float) -> Tuple[np.ndarray, np.ndarray]:
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y: np.ndarray) -> float:
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

    def _most_common_label(self, y: np.ndarray) -> Any:
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def _traverse_tree(self, x: np.ndarray, node: Node) -> Any:
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)