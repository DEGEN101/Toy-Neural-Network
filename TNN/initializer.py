import abc
import numpy as np


class Initializer(abc.ABC):
    @abc.abstractmethod
    def generate(in_features: int, out_features: int) -> np.ndarray:
        pass


class UniformXavier(Initializer):
    def generate(in_features: int, out_features: int) -> np.ndarray:
        x = np.sqrt(6.0 / (in_features + out_features))
        return np.random.uniform(-x, x, size=[out_features, in_features + 1])


class NormalXavier(Initializer):
    def generate(in_features: int, out_features: int) -> np.ndarray:
        mu = 0
        sigma = np.sqrt(2.0 / (in_features + out_features))
        return np.random.normal(mu, sigma, size=[out_features, in_features + 1])


class UniformHe(Initializer):
    def generate(in_features: int, out_features: int) -> np.ndarray:
        low = -np.sqrt(6.0 / in_features)
        high = np.sqrt(6.0 / out_features)
        return np.random.uniform(low, high, size=[out_features, in_features + 1])


class NormalHe(Initializer):
    def generate(in_features: int, out_features: int) -> np.ndarray:
        mu = 0
        sigma = np.sqrt(2.0 / (in_features))
        return np.random.normal(mu, sigma, size=[out_features, in_features + 1])
