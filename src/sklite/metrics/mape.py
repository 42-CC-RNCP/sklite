import numpy as np
from sklite.core.metric import Metric


class MeanAbsolutePercentageError(Metric):
    """
    Mean Absolute Percentage Error (MAPE) metric.

    This class computes the Mean Absolute Percentage Error between true and predicted values.
    """

    def __init__(self):
        super().__init__()
        self._name = 'Mean Absolute Percentage Error'
        self._description = 'Mean Absolute Percentage Error (MAPE) metric.'

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the Mean Absolute Percentage Error.

        Parameters
        ----------
        y_true : np.ndarray
            True values.
        y_pred : np.ndarray
            Predicted values.

        Returns
        -------
        float
            The Mean Absolute Percentage Error.
        """
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# Example usage
if __name__ == "__main__":
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])

    mape_metric = MeanAbsolutePercentageError()
    mape_value = mape_metric(y_true, y_pred)

    print(f"Mean Absolute Percentage Error: {mape_value}")
