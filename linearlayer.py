import numpy as np

from activation import Activation


def initialize_weigths(in_features : int, out_features : int, init_method : str) -> np.ndarray:
    if init_method == "xavier":
        x = np.sqrt(6.0 / (in_features + out_features))
        return np.random.uniform(-x, x, size=[out_features, in_features + 1])


class LinearLayer:
    weights: np.ndarray = None
    activation: Activation = None

    def __init__(self, in_features : int, out_features : int, activation_: Activation, init_method : str = "xavier"):
        self.weights = initialize_weigths(in_features, out_features, init_method)
        self.activation = activation_()

    def forward(self, x : np.ndarray) -> np.ndarray:
        x_ = np.vstack([np.array([1]), x])
        z = self.weights @ x_
        a = self.activation.function(z)
        return a

    def __call__(self, x : np.ndarray) -> np.ndarray:
        return self.forward(x)

    def get_weights(self) -> np.ndarray:
        return self.weights

    def set_weights(self, new_weights : np.ndarray) -> None:
        self.weights = new_weights.copy()
