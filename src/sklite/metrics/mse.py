import numpy as np
from sklite.core.metric import Metric


class MeanSquaredError(Metric):
    """
    Mean Squared Error (MSE) metric.

    This class computes the Mean Squared Error between true and predicted values.
    """

    def __init__(self):
        super().__init__()
        self._name = 'Mean Squared Error'
        self._description = 'Mean Squared Error (MSE) metric.'

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the Mean Squared Error.

        Parameters
        ----------
        y_true : np.ndarray
            True values.
        y_pred : np.ndarray
            Predicted values.

        Returns
        -------
        float
            The Mean Squared Error.
        """
        return np.mean((y_true - y_pred) ** 2)


# Example usage
if __name__ == "__main__":
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])

    mse_metric = MeanSquaredError()
    mse_value = mse_metric(y_true, y_pred)

    print(f"Mean Squared Error: {mse_value}")

