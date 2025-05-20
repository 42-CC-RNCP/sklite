import numpy as np
import pandas as pd
from typing import Tuple, Generator, Union, List


def _to_numpy(X, y):
    X_np = X.to_numpy() if isinstance(X, pd.DataFrame) else np.array(X)
    y_np = y.to_numpy() if isinstance(y, pd.Series) else np.array(y) if y is not None else None
    return X_np, y_np


def train_val_split(
    X: Union[np.ndarray, pd.DataFrame, List],
    y: Union[np.ndarray, pd.Series, List, None] = None,
    val_size: float = 0.2,
    shuffle: bool = True,
    random_state: Union[int, None] = None,
) -> Tuple[np.ndarray, Union[np.ndarray, None], np.ndarray, Union[np.ndarray, None]]:
    X_np, y_np = _to_numpy(X, y)
    n_samples = len(X_np)
    indices = np.arange(n_samples)

    if shuffle:
        rng = np.random.default_rng(seed=random_state)
        rng.shuffle(indices)

    X_np = X_np[indices]
    y_np = y_np[indices] if y is not None else None

    split_index = int(n_samples * (1 - val_size))

    return (
        X_np[:split_index],
        y_np[:split_index] if y is not None else None,
        X_np[split_index:],
        y_np[split_index:] if y is not None else None,
    )


def train_test_split(
    X: Union[np.ndarray, pd.DataFrame, List],
    y: Union[np.ndarray, pd.Series, List, None] = None,
    test_size: float = 0.2,
    shuffle: bool = True,
    random_state: Union[int, None] = None,
) -> Tuple[np.ndarray, Union[np.ndarray, None], np.ndarray, Union[np.ndarray, None]]:
    X_np, y_np = _to_numpy(X, y)
    n_samples = len(X_np)
    indices = np.arange(n_samples)

    if shuffle:
        rng = np.random.default_rng(seed=random_state)
        rng.shuffle(indices)

    X_np = X_np[indices]
    y_np = y_np[indices] if y is not None else None

    split_index = int(n_samples * (1 - test_size))

    return (
        X_np[:split_index],
        y_np[:split_index] if y is not None else None,
        X_np[split_index:],
        y_np[split_index:] if y is not None else None,
    )


def train_val_test_split(
    X: Union[np.ndarray, pd.DataFrame, List],
    y: Union[np.ndarray, pd.Series, List, None] = None,
    val_size: float = 0.2,
    test_size: float = 0.2,
    shuffle: bool = True,
    random_state: Union[int, None] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if val_size + test_size >= 1.0:
        raise ValueError("val_size + test_size must be less than 1.0")
    
    X_np, y_np = _to_numpy(X, y)
    n_samples = len(X_np)
    indices = np.arange(n_samples)

    if shuffle:
        rng = np.random.default_rng(seed=random_state)
        rng.shuffle(indices)

    X_np = X_np[indices]
    y_np = y_np[indices] if y is not None else None

    n_test = int(n_samples * test_size)
    n_val = int(n_samples * val_size)
    n_train = n_samples - n_test - n_val
    train_end = n_train
    val_end = n_train + n_val

    X_train, X_val, X_test = X_np[:train_end], X_np[train_end:val_end], X_np[val_end:]
    
    y_train = y_np[:train_end] if y is not None else None
    y_val = y_np[train_end:val_end] if y is not None else None
    y_test = y_np[val_end:] if y is not None else None
    return X_train, y_train, X_val, y_val, X_test, y_test


def kfold_split(
    X: Union[np.ndarray, pd.DataFrame, List],
    y: Union[np.ndarray, pd.Series, List, None] = None,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: Union[int, None] = None,
) -> Generator[Tuple[np.ndarray, Union[np.ndarray, None], np.ndarray, Union[np.ndarray, None]], None, None]:
    X_np, y_np = _to_numpy(X, y)
    n_samples = len(X_np)
    indices = np.arange(n_samples)

    if shuffle:
        rng = np.random.default_rng(seed=random_state)
        rng.shuffle(indices)

    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[:n_samples % n_splits] += 1

    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        val_idx = indices[start:stop]
        train_idx = np.concatenate((indices[:start], indices[stop:]))

        X_train, X_val = X_np[train_idx], X_np[val_idx]
        if y_np is not None:
            y_train, y_val = y_np[train_idx], y_np[val_idx]
            yield X_train, y_train, X_val, y_val
        else:
            yield X_train, None, X_val, None
        current = stop


def stratified_kfold_split(
    X: Union[np.ndarray, pd.DataFrame, List],
    y: Union[np.ndarray, pd.Series, List],
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: Union[int, None] = None,
) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None, None]:
    X_np, y_np = _to_numpy(X, y)
    n_samples = len(X_np)
    classes = np.unique(y_np)
    rng = np.random.default_rng(seed=random_state)

    class_indices = {cls: np.where(y_np == cls)[0] for cls in classes}
    if shuffle:
        for cls in class_indices:
            rng.shuffle(class_indices[cls])

    folds = [[] for _ in range(n_splits)]
    for cls in classes:
        indices = class_indices[cls]
        fold_sizes = np.full(n_splits, len(indices) // n_splits, dtype=int)
        fold_sizes[:len(indices) % n_splits] += 1

        current = 0
        for i, fold_size in enumerate(fold_sizes):
            folds[i].extend(indices[current:current + fold_size])
            current += fold_size

    for fold in folds:
        val_idx = np.array(fold)
        train_idx = np.setdiff1d(np.arange(n_samples), val_idx)
        yield X_np[train_idx], y_np[train_idx], X_np[val_idx], y_np[val_idx]
    