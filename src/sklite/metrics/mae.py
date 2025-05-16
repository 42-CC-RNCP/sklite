import numpy as np
from sklite.core.metric import Metric


class MeanAbsoluteError(Metric):
    """
    Mean Absolute Error (MAE) metric.

    This class computes the Mean Absolute Error between true and predicted values.
    """

    def __init__(self):
        super().__init__()
        self._name = 'Mean Absolute Error'
        self._description = 'Mean Absolute Error (MAE) metric.'

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the Mean Absolute Error.

        Parameters
        ----------
        y_true : np.ndarray
            True values.
        y_pred : np.ndarray
            Predicted values.

        Returns
        -------
        float
            The Mean Absolute Error.
        """
        return np.mean(np.abs(y_true - y_pred))


# Example usage
if __name__ == "__main__":
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])

    mae_metric = MeanAbsoluteError()
    mae_value = mae_metric(y_true, y_pred)

    print(f"Mean Absolute Error: {mae_value}")
