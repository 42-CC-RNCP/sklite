import pytest
import numpy as np
from sklite.metrics import *


@pytest.fixture
def sample_data():
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([1, 0, 0, 1])
    return y_true, y_pred


def test_confusion_matrix(sample_data):
    y_true, y_pred = sample_data
    confusion_matrix_metric = ConfusionMatrix()
    confusion_matrix_value = confusion_matrix_metric(y_true, y_pred)

    expected_confusion_matrix = np.array([[1, 0], [1, 2]])  # True Negatives, False Positives, False Negatives, True Positives
    assert np.array_equal(confusion_matrix_value, expected_confusion_matrix), f"Expected {expected_confusion_matrix}, but got {confusion_matrix_value}"
    assert isinstance(confusion_matrix_value, np.ndarray), "Confusion matrix value should be a numpy array"
    assert confusion_matrix_metric._name == 'Confusion Matrix', "Metric name should be 'Confusion Matrix'"
    assert confusion_matrix_metric._description == 'Confusion Matrix metric.', "Metric description should be 'Confusion Matrix metric.'"
