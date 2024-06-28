import numpy as np

from typing import List, Optional

from initializer import NormalHe
from linearlayer import LinearLayer
from activation import Tanh, ReLU
from optimizer import SGD, RMSProp


class NeuralNetwork:
    parameters: List[LinearLayer] = None

    def __init__(self, parameters: Optional["LinearLayer"] = None):
        self.parameters = parameters
    
    def predict(self, x : np.ndarray) -> np.ndarray:
        result = x.transpose()
        for parameter in self.parameters:
            result = parameter(result)
        return result

    def add_parameter(self, parameter):
        if self.parameters == None: self.parameters = []
        self.parameters.append(parameter)

    def get_parameters(self) -> List[LinearLayer]:
        return self.parameters


def main():
    model = NeuralNetwork()
    model.add_parameter(LinearLayer(2, 8, ReLU, init_method=NormalHe))
    model.add_parameter(LinearLayer(8, 8, ReLU, init_method=NormalHe))
    model.add_parameter(LinearLayer(8, 1, Tanh))

    optimizer = RMSProp(model.get_parameters(), 0.01)

    x = np.array([[[0, 0]], [[1, 0]], [[0, 1]], [[1, 1]]])
    y = np.array([[[0]], [[1]], [[1]], [[0]]])

    for i in range(x.shape[0]):
        y_pred = model.predict(x[i])
        print(f"Input: {x[i]}, Prediction: {y_pred}, Actual: {y[i]}")
    
    print("[!] Training Model")
    for _ in range(250):
        for i in range(x.shape[0]):
            y_pred = model.predict(x[i])
            loss = y_pred - y[i]

            optimizer.step(loss)
    print("[+] Done Training")

    for i in range(x.shape[0]):
        y_pred = model.predict(x[i])
        print(f"Input: {x[i]}, Prediction: {y_pred}, Actual: {y[i]}")

if __name__ == "__main__":
    main()
