import numpy as np
from typing import Generator, Tuple
from .base import Splitter
from .function import train_val_test_split


class TrainValTestSplitter(Splitter):
    def __init__(self, val_size=0.2, test_size=0.2, shuffle=True, random_state=None):
        self.val_size = val_size
        self.test_size = test_size
        self.shuffle = shuffle
        self.random_state = random_state
        
    def split(self, X: np.ndarray, y: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return train_val_test_split(
            X,
            y,
            self.val_size,
            self.test_size,
            self.shuffle,
            self.random_state
        )


# Example usage
if __name__ == "__main__":
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([0, 1, 0, 1, 0])
    
    splitter = TrainValTestSplitter(val_size=0.2, test_size=0.2)
    X_train, y_train, X_val, y_val, X_test, y_test = splitter.split(X, y)
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_val:", X_val.shape)
    print("y_val:", y_val.shape)
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)
