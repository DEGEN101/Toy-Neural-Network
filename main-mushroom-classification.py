import numpy as np

from typing import List, Optional

from TNN.initializer import NormalHe
from TNN.linearlayer import LinearLayer
from TNN.activation import Sigmoid, ReLU
from TNN.optimizer import RMSProp
from TNN.losses import CategoricalCrossEntropyLoss


class NeuralNetwork:
    parameters: List[LinearLayer] = None

    def __init__(self, parameters: Optional["LinearLayer"] = None):
        self.parameters = parameters
    
    def predict(self, x : np.ndarray) -> np.ndarray:
        result = x.reshape(-1, *x.shape).transpose()
        for parameter in self.parameters:
            result = parameter(result)
        return result

    def add_parameter(self, parameter):
        if self.parameters == None: self.parameters = []
        self.parameters.append(parameter)

    def get_parameters(self) -> List[LinearLayer]:
        return self.parameters


def build_model() -> NeuralNetwork:
    model = NeuralNetwork()
    
    model.add_parameter(LinearLayer(2, 8, ReLU, init_method=NormalHe))
    model.add_parameter(LinearLayer(8, 8, ReLU, init_method=NormalHe))
    model.add_parameter(LinearLayer(8, 2, Sigmoid))

    return model


def load_data() -> List[np.ndarray]:
    pass 


def main():
    model = build_model()
    optimizer = RMSProp(model.get_parameters(), 0.01)
    criterion = CategoricalCrossEntropyLoss()
    

if __name__ == "__main__":
    main()
