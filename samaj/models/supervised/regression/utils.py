from typing import List, Tuple

import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn import metrics


def record_learning_curves(
    model: SGDRegressor,
    X_train: np.array,
    y_train: np.array,
    X_test: np.array,
    y_test: np.array,
    num_epochs=1000,
) -> Tuple[List[float], List[float]]:
    """
    Records the learning curves from a model training.
    This is intended to capture the train/test skew of
    models that train via minimizing a loss function, e.g. SGD.
    Parameters:
    model(SGDRegressor): the estimator
    X_train, y_train, X_test, y_test (np.array): the data sets
    num_epochs(int)
    Returns: tuple: two lists of the model's MSE in each epoch,
             for both training and test data
    """
    train_errors, test_errors = list(), list()
    for _ in range(num_epochs):
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        train_errors.append(metrics.mean_squared_error(y_train, y_train_pred))
        test_errors.append(metrics.mean_squared_error(y_test, y_test_pred))
    return train_errors, test_errors