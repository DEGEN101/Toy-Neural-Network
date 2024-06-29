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


def main():
    model = NeuralNetwork()
    model.add_parameter(LinearLayer(2, 8, ReLU, init_method=NormalHe))
    model.add_parameter(LinearLayer(8, 8, ReLU, init_method=NormalHe))
    model.add_parameter(LinearLayer(8, 2, Sigmoid))

    optimizer = RMSProp(model.get_parameters(), 0.01)
    criterion = CategoricalCrossEntropyLoss()

    x = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

    for i in range(x.shape[0]):
        y_pred = model.predict(x[i])
        print(f"Input: {x[i]}, Prediction: {y_pred.argmax()}, Actual: {y[i].argmax()}")
    
    print("[!] Training Model")
    for epoch in range(1, 150 + 1):
        loss = np.zeros(shape=y.shape)

        for i in range(x.shape[0]):
            y_pred = model.predict(x[i])
            y_true = y[i].reshape(-1, *y[i].shape).transpose()

            loss += criterion.function(y_true, y_pred)
            loss_grad = criterion.derivative(y_true, y_pred)

            optimizer.step(loss_grad)

        if epoch % 25 == 0:
            print(f"Epoch: {epoch}, Loss: {loss.mean()}")

    print("[+] Done Training")

    for i in range(x.shape[0]):
        y_pred = model.predict(x[i])
        print(f"Input: {x[i]}, Prediction: {y_pred.argmax()}, Actual: {y[i].argmax()}")

if __name__ == "__main__":
    main()
