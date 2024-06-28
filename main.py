import numpy as np

from linearlayer import LinearLayer
from activation import Sigmoid


def main():
    theta_1 = np.array([[-2, 3], [2, -1]], dtype=np.float32)
    theta_2 = np.array([2, -1, 1], dtype=np.float32)

    l1 = LinearLayer(1, 2, Sigmoid)
    l1.set_weights(theta_1)

    l2 = LinearLayer(2, 1, Sigmoid)
    l2.set_weights(theta_2)

    x = np.array([2])
    print(output := l1(x))
    print(l2(output))

if __name__ == "__main__":
    main()
