# ******************************************************************************
# PROMPT
# ------
# Given a set of 2D data points, compute the polynomial that
# best fits the data.
# **************************************

from sklearn import preprocessing
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
from numpy import random

from samaj.models.supervised import base

# ==============================================================================
# LOAD DATA
# ==============================================================================
# Load and separate the data
domain = np.arange(100)
noise = random.rand(100)
target = (2 * (domain ** 2)) + (5 * domain) + noise

# Split the data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    domain, target, test_size=0.20, random_state=42
)

# ==============================================================================
# PROBLEM: Find the best fit polynomial
#
# APPROACH 1: Linear Regression - this is an example of a model that does NOT
#             use a closed form solution, and instead relies on backprop through time.
# ==============================================================================
class LinearRegressorBackprop(base.BaseModel):
    def __init__(
        self,
        evaluate_on={
            "MSE": make_scorer(mean_squared_error),
            "R2": make_scorer(r2_score),
        },
        include_bias=True
    ):
        super().__init__(evaluate_on)
        self.parameters = list()  #  default value, because we don't know how many features yet
        if include_bias is True:
            self.intercept = 0

    def fit(self, X_train: np.array, y_train: np.array, 
            epochs: int, optimizer):
        """
        Goal is to find the best parameters, via minimizing the loss.

        X_train, y_train: arrays with shapes are (m, n) and (m, 1) respectively.
        epochs: int
        optimizer: should have a learning rate, and a tolerance at least

        Returns: None. Updates the parameter + intercept state at the end of training loop.
        """
        # formulate the weight matrix to pass to the optimizer
        num_samples, num_features = X_train.shape

        weight_vector = random.rand(num_features).reshape(num_features, 1)  # col vector
        if hasattr(self, "intercept") is True:
            weight_vector = np.concat([weight_vector, np.ones(1, 1)], 0)

        # let the optimizer converge
        for _ in range(epochs):
            y_pred = np.dot(X_train, weight_vector)
            error = y_pred - y_train
            weight_vector = optimizer.compute_weight_update(
                error, "MSE", m=num_samples, X_train=X_train
            )

        # and set the best new params
        best_params = np.squeeze(weight_vector.reshape(1, -1))  # 1D vector
        if hasattr(self, "intercept"):
            self.intercept = best_params[-1]
            self.parameters = best_params[:-1]
        else:
            self.parameters = best_params[:]

    def predict(self, X_test):
            weight_vector = np.array([self.parameters])
            if hasattr(self, "intercept") is True:
                weight_vector = np.concat([weight_vector, np.ones(1, 1)], 0)
            y_pred = X_test.dot(weight_vector.T)
            return y_pred


if __name__ == "__main__":
    num_cross_val_folds = 5
    print("==========================================================")
    print("Example: LINEAR REGRESSION, using Backpropagation", end="\n\n")
    print(
        f"training/testing the model with {num_cross_val_folds}-fold cross validation...",
        end="\n\n",
    )
    # TODO
    pass
