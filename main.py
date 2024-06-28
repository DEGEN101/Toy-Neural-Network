import numpy as np

from linearlayer import LinearLayer
from activation import Sigmoid
from optimizer import SGD

def main():
    theta_1 = np.array([[-2, 3], [2, -1]], dtype=np.float32)
    theta_2 = np.array([[2, -1, 1]], dtype=np.float32)

    l1 = LinearLayer(1, 2, Sigmoid)
    l1.set_weights(theta_1)

    l2 = LinearLayer(2, 1, Sigmoid)
    l2.set_weights(theta_2)

    layers = [l1, l2]
    optimizer = SGD(layers, 0.2)

    x, y = np.array([2]), np.array([4])
    for layer in layers:
        x = layer(x)
    
    error = x - y
    print(error)

    optimizer.step(error)

    for layer in layers:
        print(layer.get_weights())

if __name__ == "__main__":
    main()
