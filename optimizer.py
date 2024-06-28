import abc
import numpy as np

from typing import List
from copy import deepcopy

from linearlayer import LinearLayer


class Optimizer(abc.ABC):
    gradients: List[np.ndarray] = None
    parameters: List[LinearLayer] = None
    alpha: float = None

    def __init__(self, parameters_ : List[LinearLayer], learning_rate : float):
        self.parameters = parameters_
        self.alpha = learning_rate
    
        self.zero_gradients()

    def zero_gradients(self) -> None:
        self.gradients = []

        for parameter in self.parameters:
            gradient = np.zeros(shape=parameter.get_weights().shape)
            self.gradients.append(gradient)

    def compute_error(self, curr_param: LinearLayer, next_param: LinearLayer, next_error: np.ndarray) -> np.ndarray:
        sub_weights = next_param.get_weights()[:, 1:]

        return sub_weights.transpose() @ next_error * curr_param.activation.derivative(curr_param.z)
    
    @abc.abstractmethod
    def step(self, loss : np.ndarray) -> None:
        raise NotImplementedError()


class SGD(Optimizer):
    def __init__(self, parameters_ : List[LinearLayer], learning_rate : float):
        super().__init__(parameters_, learning_rate)

    def step(self, loss : np.ndarray) -> None:
        N = len(self.parameters)

        errors = [np.array([]) for _ in range(N)]
        errors[N - 1] = loss
        
        for i in range(N - 2, -1, -1):
            errors[i] = self.compute_error(self.parameters[i], self.parameters[i + 1], errors[i + 1])

        for i in range(N):
            self.gradients[i] = errors[i] @ self.parameters[i].input.transpose()
            
            dtheta = self.alpha * self.gradients[i]
            new_weights = self.parameters[i].get_weights() - dtheta
            self.parameters[i].set_weights(new_weights)


class RMSProp(Optimizer):
    beta: float = None
    epsilon: float = None
    squared_gradients : List[np.ndarray] = None

    def __init__(self, parameters : List[LinearLayer], learning_rate : float, *, decay_rate : float = 0.9, epsilon_ : float = 1e-8):
        super().__init__(parameters, learning_rate)

        self.beta = decay_rate
        self.epsilon = epsilon_
        self.squared_gradients = deepcopy(self.parameters)
    
    def step(self, loss : np.ndarray) -> None:
        pass

class RMSProp(Optimizer):
    def __init__(self, parameters : List[LinearLayer], learning_rate : float, decay_rate : float):
        super().__init__(parameters, learning_rate)
    
    def step(self, loss : np.ndarray):
        pass
