import numpy as np


def sigmoid(z) -> float:
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative(z) -> float:
    return sigmoid(z) * (1.0 - sigmoid(z))


def relu(z):
    pass

def relu_derivative(z):
    pass


class Activation:
    function = None
    derivative = None

    def __init__(self, f, d):
        self.function = f
        self.derivative = d   


class Sigmoid(Activation):
    def __init__(self):
        super(Sigmoid, self).__init__(sigmoid, sigmoid_derivative)


class ReLU(Activation):
    def __init__(self):
        super(ReLU, self).__init__(relu, relu_derivative)
