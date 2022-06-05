import numpy as np


class MSE:
    """Mean-squared error, a cost function commonly used in regression."""
    def __init__(self):
        self.name = "mse"

    def __call__(self, y_pred: np.array, y_true: np.array):
        sum_squared_errors = np.sum((y_pred - y_true) ** 2)
        num_samples = np.squeeze(y_pred).shape
        return sum_squared_errors / num_samples
