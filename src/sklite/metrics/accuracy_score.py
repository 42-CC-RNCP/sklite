import numpy as np
from sklite.core.metric import Metric


class AccuracyScore(Metric):
    """
    Accuracy Score metric.

    This class computes the accuracy score between true and predicted values.
    """

    def __init__(self):
        super().__init__()
        self._name = 'Accuracy Score'
        self._description = 'Accuracy Score metric.'

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the accuracy score.

        Parameters
        ----------
        y_true : np.ndarray
            True values.
        y_pred : np.ndarray
            Predicted values.

        Returns
        -------
        float
            The accuracy score.
        """
        return np.mean(y_true == y_pred) if len(y_true) > 0 else 0.0


# Example usage
if __name__ == "__main__":
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([1, 0, 0, 1])

    accuracy_metric = AccuracyScore()
    accuracy_value = accuracy_metric(y_true, y_pred)

    print(f"Accuracy Score: {accuracy_value}")
