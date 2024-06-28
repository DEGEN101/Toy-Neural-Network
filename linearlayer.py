import numpy as np


class LinearLayer:
    def __init__(self, in_features : int, out_features : int, init_method : str):
        pass

    def forward(self, x : np.ndarray) -> np.ndarray:
        pass

    def __call__(self, x : np.ndarray) -> np.ndarray:
        pass

    def get_weights(self):
        pass

    def update_weights(self, new_weights : np.ndarray):
        pass
