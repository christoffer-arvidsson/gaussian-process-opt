"""Module for kernel functions."""
import numpy as np
import numpy.typing as npt
import scipy


class RBFKernel:
    """Exponentiated quadratic covariance function."""
    def __init__(self, signal_length_factor: float = 1.0):
        self.length = signal_length_factor

    def __call__(self, x_a: npt.NDArray[np.float32], x_b: npt.NDArray[np.float32]):
        return np.exp(-(1 / (2 * self.length**2)) * scipy.spatial.distance.cdist(x_a, x_b, 'sqeuclidean'))

class PeriodicKernel:
    """Periodic kernel."""
    def __init__(self, signal_length_factor: float = 1.0, frequency: float = np.pi):
        self._length = signal_length_factor
        self._freq = frequency

    def __call__(self, x_a: npt.NDArray[np.float32], x_b: npt.NDArray[np.float32]):
        dist = scipy.spatial.distance.cdist(x_a, x_b, 'euclidean')
        return np.exp(-(2 * np.sin(np.pi * dist/self._freq)**2 / self._length**2))

class LinearKernel:
    """Linear kernel."""
    def __init__(self, offset: float = 0.0, std_offset: float=1.0, std: float= 1.0):
        self._offset = offset
        self._std_offset = std_offset
        self._std = std

    def __call__(self, x_a: npt.NDArray[np.float32], x_b: npt.NDArray[np.float32]):
        return self._std_offset**2 + self._std**2 * ((x_a-self._offset) @ (x_b-self._offset).T)

class WhiteNoiseKernel:
    """Kernel intended for noise."""
    def __init__(self, std: float = 1.0):
        self._std = std

    def __call__(self, x_a: npt.NDArray[np.float32], x_b: npt.NDArray[np.float32]):
        return self._std ** 2 * np.eye(x_a.shape[0])
