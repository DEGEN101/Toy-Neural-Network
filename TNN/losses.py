import numpy as np


class Loss:
    function = None
    derivative = None

    def __init__(self, function, derivative):
        self.function = function
        self.derivative = derivative


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.sum((y_pred - y_true) ** 2) / np.size(y_true)

def mse_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return (2 / np.size(y_true)) * (y_pred - y_true)

class MSELoss(Loss):
    def __init__(self):
        super(MSELoss, self).__init__(mse, mse_derivative)


def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    epsilon = 1e-15 
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return - np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / np.size(y_true)

def binary_cross_entropy_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    epsilon = 1e-15 
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return (y_pred - y_true) / (y_pred * (1 - y_pred))

class BinaryCrossEntropy(Loss):
    def __init__(self):
        super(BinaryCrossEntropy, self).__init__(binary_cross_entropy, binary_cross_entropy_derivative)


def categorical_cross_entropy(y_true, y_pred):
    epsilon = 1e-15  # to prevent log(0) error
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # clip predictions to prevent log(0)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=-1), axis=-1)

def categorical_cross_entropy_derivative(y_true, y_pred):
    return (y_pred - y_true) / len(y_true)

class CategoricalCrossEntropyLoss(Loss):
    def __init__(self):
        super(CategoricalCrossEntropyLoss, self).__init__(categorical_cross_entropy, categorical_cross_entropy_derivative)
