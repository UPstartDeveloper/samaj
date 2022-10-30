from dataclasses import dataclass
import math

import numpy as np
from numpy.random import default_rng, Generator
from sklearn.metrics import make_scorer, accuracy_score

from samaj.models.supervised import base
from samaj.util import activations


@dataclass
class BinaryClassificationMLP(base.BaseModel):
    """Only supports 2-layer networks for the moment."""
    num_layers: int
    units_per_layer: np.ndarray
    threshold: float = 0.5
    classes: np.ndarray = np.array([0, 1])
    rng: Generator = default_rng(seed=42)
    evaluate_on: dict = {"Accuracy": make_scorer(accuracy_score)}

    def __init__(self, *args, **kwargs):
        super().__init__(self.evaluate_on)

    def define_model(self, num_features: int) -> None:
        """Glorot weight initialization"""
        self.layers = list()
        fan_in = num_features
        fan_out_index = 0
        for layer_index in range(self.num_layers):
            # initialize using a randomly sampled uniform distribution
            fan_out = self.units_per_layer[fan_out_index]
            scale = max(1.0, (fan_in + fan_out) / 2.0)
            limit = math.sqrt(3.0 * scale)
            layer_weights = self.rng.uniform(
                low=-limit, high=limit, size=(fan_in, fan_out)
            )
            # biases - initialize to zeros, b/c we don't need to break symmetry (unlike for the weights)
            layer_bias = np.zeros((fan_out, 1))
            # activation - use tanh for hidden layers, and sigmoid for the last one
            activation = np.tanh
            if layer_index == self.num_layers - 1:
                activation = activations.sigmoid
            # add to the list, and prep for next iteration
            self.layers.append((layer_weights, layer_bias, activation))
            fan_in = fan_out
            fan_out_index += 1

    def forward(self, X: np.ndarray) -> np.ndarray:
        current_activation = X
        layer_activations = list()
        for weight, bias, act_func in self.layers:
            layer_summation = current_activation @ weight + bias.T
            current_activation = act_func(layer_summation)
            layer_activations.append(current_activation)
        return layer_activations

    def backward(self, X, y, learning_rate, activations) -> None:
        """TODO: generalize to L layers"""
        # variables we're going to need
        weights1, bias1, act1 = self.layers[0]
        weights2, bias2, act2 = self.layers[1]
        per_sample_factor = (1 / X.shape[0])
        hidden_layer_activation, output_layer_activations = (
            activations[0], activations[1]
        )
        output_layer_weights = weights2
        num_samples = X.shape[0]

        # derivatives for the output layer
        y_pred = output_layer_activations
        y_true = np.where(y == -1, 0, 1).reshape(-1, 1)  # labels should be only 0/1
        error = y_pred - y_true
        derivative_y_pred = y_pred * (1 - y_pred)
        grad_output_layer = error * derivative_y_pred
        derivative_output_layer = dW2 = (grad_output_layer.T @ hidden_layer_activation).T
        derivative_output_bias = db2 = (1 / num_samples) * np.sum(error, axis=0, keepdims=True)

        # update weights in output layer before going fwd
        new_output_weights = weights2 - learning_rate * dW2
        new_output_bias = bias2 - learning_rate * db2

        # derivatives for the hidden layer
        derivative_hidden_activation = z_prime = dZ1 = hidden_layer_activation * (1 - hidden_layer_activation)
        grad_hidden_layer = grad_output_layer @ new_output_weights.T * z_prime
        derivative_hidden_weights = X.T @ grad_hidden_layer
        derivative_hidden_bias = db1 = (1 / num_samples) * np.sum(z_prime, axis=0, keepdims=True).T

        # update weights in the hidden layer
        new_hidden_weights = weights1 - learning_rate * derivative_hidden_weights
        new_hidden_bias = bias1 - learning_rate * db1

        # update the state of the model
        self.layers[0] = (new_hidden_weights, new_hidden_bias, act1)
        self.layers[1] = (new_output_weights, new_output_bias, act2)

    def fit(
        self, X_train: np.array, y_train: np.array, epochs=1000, learning_rate=0.0001
    ) -> "BinaryClassificationMLP":
        # A: initial state of the network
        num_features = X_train.shape[1]
        self.define_model(num_features)

        layer1, layer2 = self.layers[0], self.layers[1]

        # B: training!
        for _ in range(epochs):
            activations = self.forward(X_train)
            self.backward(X_train, y_train, learning_rate, activations)
            # TODO: add callbacks/regularizers?
        
        return self

    def predict(self, X) -> np.ndarray:
        activations = self.forward(X)
        class1, class2 = self.classes
        y_pred = np.where(activations[1] >= self.threshold, class2, class1)
        return y_pred


if __name__ == "__main__":
    num_layers = 2
    units_per_layer = [5, 1]
    num_features = 2
    classes = np.array([-1, 1])
    custom_mlp = BinaryClassificationMLP(
        num_layers,
        units_per_layer,
        num_features,
    )
