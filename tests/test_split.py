import numpy as np
import pandas as pd
from sklite.split.function import (
    train_test_split,
    train_val_split,
    train_val_test_split,
    kfold_split,
    stratified_kfold_split,
)


def test_train_test_split():
    X = pd.DataFrame(np.arange(20).reshape(10, 2), columns=["a", "b"])
    y = pd.Series(np.arange(10))

    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

    assert X_train.shape[0] == 7, "Incorrect train size"
    assert X_test.shape[0] == 3, "Incorrect test size"
    assert y_train.shape[0] == 7 and y_test.shape[0] == 3, "Label size mismatch"
    assert isinstance(X_train, np.ndarray) and isinstance(y_train, np.ndarray), "Output type not ndarray"


def test_train_val_split():
    X = list(range(10))
    y = list(range(10))

    X_train, y_train, X_val, y_val = train_val_split(X, y, val_size=0.4, shuffle=False)

    assert len(X_train) == 6
    assert len(X_val) == 4
    assert all(isinstance(arr, np.ndarray) for arr in [X_train, y_train, X_val, y_val])


def test_train_val_test_split_with_labels():
    X = np.arange(30).reshape(10, 3)
    y = np.arange(10)

    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y, val_size=0.3, test_size=0.2)

    assert X_train.shape[0] == 5   # 10 - 3 - 2 = 5
    assert X_val.shape[0] == 3
    assert X_test.shape[0] == 2
    assert y_train.shape[0] == 5
    assert y_val.shape[0] == 3
    assert y_test.shape[0] == 2


def test_train_val_test_split_without_labels():
    X = np.arange(30).reshape(10, 3)

    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, val_size=0.3, test_size=0.2)

    # Total = 10
    # test_size = 0.2 → 2
    # val_size  = 0.3 → 3
    # train     = 5

    assert X_train.shape[0] == 5
    assert X_val.shape[0] == 3
    assert X_test.shape[0] == 2
    assert y_train is None and y_val is None and y_test is None


def test_kfold_split_outputs():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)

    folds = list(kfold_split(X, y, n_splits=5, shuffle=False))

    assert len(folds) == 5
    for X_train, y_train, X_val, y_val in folds:
        assert X_train.shape[0] + X_val.shape[0] == 10
        assert y_train.shape[0] + y_val.shape[0] == 10
        assert X_val.shape[1] == 2


def test_stratified_kfold_distribution():
    X = np.random.rand(20, 3)
    y = np.array([0, 1] * 10)

    folds = list(stratified_kfold_split(X, y, n_splits=4, shuffle=False))

    for X_train, y_train, X_val, y_val in folds:
        assert len(X_train) + len(X_val) == 20
        assert len(np.unique(y_val)) == 2
        assert sorted(np.unique(y_val).tolist()) == [0, 1]
