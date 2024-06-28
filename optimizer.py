import abc

from typing import Optional
from dataclasses import dataclass


import numpy as np


class Optimizer(abc.ABC):
    def __init__(self, parameters_ : np.ndarray, learning_rate : float):
        pass
    
    def zero_gradients(self) -> None:
        pass
    
    @abc.abstractmethod
    def step(self) -> None:
        raise NotImplementedError()


class SGD(Optimizer):
    def __init__(self, parameters_ : np.ndarray, learning_rate : float):
        super().__init__(parameters_, learning_rate)
        pass

    def step(self) -> None:
        pass


class RMSProp(Optimizer):
    def __init__(self, parameters_ : np.ndarray, learning_rate : float, decay_rate : float):
        super().__init__(parameters_, learning_rate)
    
    def step(self):
        pass
