import numpy as np
from sklite.core.metric import Metric


class ConfusionMatrix(Metric):
    """
    Confusion Matrix metric.

    This class computes the confusion matrix between true and predicted values.
    """

    def __init__(self):
        super().__init__()
        self._name = 'Confusion Matrix'
        self._description = 'Confusion Matrix metric.'

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate the confusion matrix.

        Parameters
        ----------
        y_true : np.ndarray
            True values.
        y_pred : np.ndarray
            Predicted values.

        Returns
        -------
        np.ndarray
            The confusion matrix.
            where rows represent true labels and columns represent predicted labels:
            (0, 0): True Negatives
            (0, 1): False Positives
            (1, 0): False Negatives
            (1, 1): True Positives
        """
        labels = np.unique(np.concatenate((y_true, y_pred)))
        n_labels = len(labels)
        confusion_matrix = np.zeros((n_labels, n_labels), dtype=int)
        for i in range(len(y_true)):
            confusion_matrix[int(y_true[i]), int(y_pred[i])] += 1
        return confusion_matrix
    

# Example usage
if __name__ == "__main__":
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([1, 0, 0, 1])

    confusion_matrix_metric = ConfusionMatrix()
    confusion_matrix_value = confusion_matrix_metric(y_true, y_pred)

    print("Confusion Matrix:")
    print(confusion_matrix_value)
