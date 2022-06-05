import numpy as np


class BatchGradientDescent:

    def __init__(self):
        self.tol = -np.inf
        self.eta0 = 0.0005  # initial learning rate

    def compute_weight_update(self, error, loss, weight_vector, **kwargs):
        # TODO: incorporate self.tol
        if loss == "MSE" and "m" in kwargs:
            m = kwargs["m"]
            X_train = kwargs["X_train"]
            gradients = 2/m * X_train.T.dot(error)

        if gradients:
            return weight_vector - self.eta0 * gradients
