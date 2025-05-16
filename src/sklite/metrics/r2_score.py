import numpy as np
from sklite.core.metric import Metric


class R2Score(Metric):
    """
    R^2 Score metric.

    This class computes the R^2 score between true and predicted values.
    """

    def __init__(self):
        super().__init__()
        self._name = 'R^2 Score'
        self._description = 'R^2 Score metric.'

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the R^2 score.

        Parameters
        ----------
        y_true : np.ndarray
            True values.
        y_pred : np.ndarray
            Predicted values.

        Returns
        -------
        float
            The R^2 score.
        """
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0


# Example usage
if __name__ == "__main__":
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])

    r2_metric = R2Score()
    r2_value = r2_metric(y_true, y_pred)

    print(f"R^2 Score: {r2_value}")
