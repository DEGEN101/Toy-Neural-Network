import abc
import numpy as np

from typing import List

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

        for parameter in self.model_parameters:
            gradient = np.zeros(shape=parameter.shape)
            self.gradients.append(gradient)

    
    @abc.abstractmethod
    def step(self, loss : np.ndarray) -> None:
        raise NotImplementedError()


class SGD(Optimizer):
    def __init__(self, parameters : List[LinearLayer], learning_rate : float):
        super().__init__(parameters, learning_rate)

    def step(self, loss : np.ndarray) -> None:
        pass


class RMSProp(Optimizer):
    def __init__(self, parameters : List[LinearLayer], learning_rate : float, decay_rate : float):
        super().__init__(parameters, learning_rate)
    
    def step(self, loss : np.ndarray):
        pass
