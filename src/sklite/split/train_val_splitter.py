import numpy as np
from typing import Tuple
from .base import Splitter
from .function import train_val_split


class TrainValSplitter(Splitter):
    """
    Train-Validation Splitter
    """

    def __init__(self, val_size=0.2, shuffle=True, random_state=None):
        self.val_size = val_size
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X: np.ndarray, y: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return train_val_split(
            X,
            y,
            self.val_size,
            self.shuffle,
            self.random_state
        )


# Example usage
if __name__ == "__main__":
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([0, 1, 0, 1, 0])

    splitter = TrainValSplitter(val_size=0.2)
    X_train, y_train, X_val, y_val = splitter.split(X, y)
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_val:", X_val.shape)
    print("y_val:", y_val.shape)
