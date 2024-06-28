import numpy as np


def sigmoid(z : np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative(z : np.ndarray) -> np.ndarray:
    return sigmoid(z) * (1.0 - sigmoid(z))


def relu(z : np.ndarray) -> np.ndarray:
    return np.maximum(z, 0)

def relu_derivative(z : np.ndarray) -> np.ndarray:
    return np.where(z > 0, 1, 0)


def tanh(z : np.ndarray) -> np.ndarray:
    return 2 / (1 + np.exp(-2 * z)) - 1

def tanh_derivative(z : np.ndarray) -> np.ndarray:
    return 1 - np.power(tanh(z), 2)
    

class Activation:
    function = None
    derivative = None

    def __init__(self, function, derivative):
        self.function = function
        self.derivative = derivative  


class Sigmoid(Activation):
    def __init__(self):
        super(Sigmoid, self).__init__(sigmoid, sigmoid_derivative)


class ReLU(Activation):
    def __init__(self):
        super(ReLU, self).__init__(relu, relu_derivative)


class Tanh(Activation):
    def __init__(self):
        super(Tanh, self).__init__(tanh, tanh_derivative)
