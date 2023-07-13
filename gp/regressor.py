"""Module for GPR."""
import numpy as np
import scipy
from typing import Callable


class GaussianProcessRegressor:
    def __init__(self, kernel_function: Callable, std_noise: float = 0.0):
        self.kernel_fn = kernel_function
        self.std_noise = std_noise

        self.x_train = None
        self.y_train = None
        self._L = None
        self._alpha = None
        self._cov_11 = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        self.x_train = x_train
        self.y_train = y_train

        epsilon = self.std_noise**2 * np.eye(x_train.shape[0])
        self._cov_11 = self.kernel_fn(x_train, x_train) + epsilon
        self._L = scipy.linalg.cholesky(self._cov_11, lower=True)
        self._alpha = scipy.linalg.cho_solve((self._L, True), y_train)

        return self

    def predict(self, x_test: np.ndarray):
        assert self._cov_11 is not None
        cov_12 = self.kernel_fn(self.x_train, x_test)
        cov_22 = self.kernel_fn(x_test, x_test)
        mu = cov_12.T @ self._alpha

        beta = np.linalg.solve(self._cov_11, cov_12)
        cov = cov_22 - cov_12.T @ beta
        std = np.sqrt(np.diag(cov))

        return mu, std, cov

    def log_marginal_likelihood(self):
        return (
            -0.5 * self.y_train.T @ self._alpha
            - np.sum(np.log(np.diag(self._L)))
            - (self._L.shape[0] / 2) * np.log(2 * np.pi)
        )
