"""Module for gaussian process optimizer."""
from typing import Callable, List
import numpy as np
import scipy
from scipy.stats import norm

from gp.regressor import GaussianProcessRegressor


class GaussianProcessOptimizer:
    def __init__(
        self,
        kernel: Callable,
        lower_bound: List[float] = [0],
        upper_bound: List[float] = [1],
        minimize: bool = False,
        acq_iterations: int = 10,
        std_noise: float = 0.0,
    ):

        assert len(lower_bound) == len(upper_bound)
        self._dims = len(lower_bound)
        self._minimize = minimize
        self._acq_iterations = acq_iterations
        self.regressor = GaussianProcessRegressor(kernel, std_noise=std_noise)
        self._x_train = np.zeros((0, self._dims))
        self._y_train = np.zeros((0,))
        self._bounds = np.array([lower_bound, upper_bound]).T

        self._opt_x = None
        self._opt_y = np.inf if self._minimize else -np.inf

    def _acquisition(self, x):
        """Expected improvement."""

        if len(x.shape) == 1:
            x = x[None, :]

        mu, sigma, cov = self.regressor.predict(x)
        flip = np.power(-1, self._minimize)
        z = flip * (mu - self._opt_y) / sigma

        out = flip * (mu - self._opt_y) * norm.cdf(z) + sigma * norm.pdf(z)

        return out


    def add_point(self, x: np.ndarray, y: float):
        self._x_train = np.append(self._x_train, x, axis=0)
        self._y_train = np.append(self._y_train, y, axis=0)

        if self._minimize:
            if y < self._opt_y:
                self._opt_x = x
                self._opt_y = y
        else:
            if y > self._opt_y:
                self._opt_x = x
                self._opt_y = y

    def random_point(self):
        return np.random.uniform(self._bounds.T[0], self._bounds.T[1], size=(1, self._dims))

    def step(self):
        self.regressor.fit(self._x_train, self._y_train)

        x = self.random_point()

        best_score = np.inf
        for i in range(self._acq_iterations):
            x0 = self.random_point()[0]
            res = scipy.optimize.minimize(
                lambda x: -self._acquisition(x),
                x0 = x0,
                method="L-BFGS-B",
                bounds=self._bounds,
            )

            if res.fun < best_score:
                best_score = res.fun
                x = res.x

        if len(x.shape) == 1:
            x = x[None, :]

        return x

    def optimize(self, obj_function: Callable, num_iterations: int = 10):
        # Add an initial random point
        x_init = self.random_point()
        y_init = obj_function(x_init)
        self.add_point(x_init, y_init)

        for i in range(num_iterations-1):
            x_prop = self.step()
            # Expensive usually
            y_prop = obj_function(x_prop)

            self.add_point(x_prop, y_prop)

        _ = self.step()


        return self._opt_x, self._opt_y
