import numpy as np
from sklite.core.metric import Metric


class RootMeanSquaredError(Metric):
    """
    Root Mean Squared Error (RMSE) metric.

    This class computes the Root Mean Squared Error between true and predicted values.
    """

    def __init__(self):
        super().__init__()
        self._name = 'Root Mean Squared Error'
        self._description = 'Root Mean Squared Error (RMSE) metric.'

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the Root Mean Squared Error.

        Parameters
        ----------
        y_true : np.ndarray
            True values.
        y_pred : np.ndarray
            Predicted values.

        Returns
        -------
        float
            The Root Mean Squared Error.
        """
        return np.sqrt(np.mean((y_true - y_pred) ** 2))


# Example usage
if __name__ == "__main__":
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])

    rmse_metric = RootMeanSquaredError()
    rmse_value = rmse_metric(y_true, y_pred)

    print(f"Root Mean Squared Error: {rmse_value}")
