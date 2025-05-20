import numpy as np
from typing import Generator, Tuple
from .base import Splitter
from .function import kfold_split


class KFoldSplitter(Splitter):
    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    
    def split(self, X: np.ndarray, y: np.ndarray = None) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None, None]:
        return kfold_split(
            X, y,
            self.n_splits,
            self.shuffle,
            self.random_state
        )


# Example usage
if __name__ == "__main__":
    import pandas as pd
    from sklearn.datasets import load_iris

    # Load sample data
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Create a KFoldSplitter instance
    splitter = KFoldSplitter(n_splits=3, shuffle=True, random_state=42)

    # Generate splits
    for X_train, X_test, y_train, y_test in splitter.split(X, y):
        print("Train shape:", X_train.shape)
        print("Test shape:", X_test.shape)
        print("Train labels shape:", y_train.shape)
        print("Test labels shape:", y_test.shape)
        print("-" * 40)
