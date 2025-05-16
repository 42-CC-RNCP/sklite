import numpy as np


class Metric:
    """
    Base class for all metrics.
    """

    def __init__(self):
        self._name = 'Metric'
        self._description = 'Base class for all metrics.'

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        raise NotImplementedError("Subclasses should implement this method.")

    def __str__(self):
        return f"{self._name}: {self._description}"

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self._name}, description={self._description})"