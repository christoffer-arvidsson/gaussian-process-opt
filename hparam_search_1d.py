import numpy as np
from numpy.typing import ArrayLike

from gp.kernel import RBFKernel
from gp.optimizer import GaussianProcessOptimizer
import matplotlib.pyplot as plt


def plot_landscape(
    x_train, y_train, x_test, y_test, mu_post, std_post, ax, title="", xlim=[-1, 1]
):
    ax.set_xlabel("input, x")
    ax.set_ylabel("output, f(x)")
    ax.set_title(title)
    ax.set_xlim(*xlim)
    ax.plot(x_test, mu_post, label="Mean", linewidth=2)
    ax.plot(
        x_test, true_function(x_test), label="True function", linewidth=5, alpha=0.5
    )

    if x_train is not None and y_train is not None:
        ax.scatter(x_train, y_train, marker="+", s=100, label="train point")

    ax.fill_between(
        x_test.reshape(-1), mu_post - 2 * std_post, mu_post + 2 * std_post, alpha=0.1
    )
    for y in y_test:
        ax.plot(x_test, y, "--", alpha=0.2, c="black")

    ax.legend()


def true_function(x: np.ndarray) -> float:
    return np.sin(2.0 * x[:, 0] - 2.0) * np.cos(0.5 * x[:, 0] + 3.0)


def main():
    n_test = 75  # num points to posterio
    n_post_functions = 5  # num functions to sample from posterior

    domain = (-5, 5) # Domain to search
    x_test = np.linspace(*domain, n_test)[:, None]


    optimizer = GaussianProcessOptimizer(
        kernel=RBFKernel(),
        lower_bound=[domain[0]],
        upper_bound=[domain[1]],
        std_noise=0.1,
        minimize=False,
    )
    opt_x, opt_y = optimizer.optimize(true_function, num_iterations=10)
    print(f"Optimal point: {opt_x}, {opt_y}")

    mu_post, std_post, cov_post = optimizer.regressor.predict(x_test)
    ys_test = np.random.multivariate_normal(
        mean=mu_post, cov=cov_post, size=n_post_functions
    )

    _, ax_post = plt.subplots(1, 1, figsize=(8, 4))
    plot_landscape(
        optimizer._x_train,
        optimizer._y_train,
        x_test,
        ys_test,
        mu_post,
        std_post,
        ax_post,
        title="posterior",
        xlim=domain,
    )
    for i, (x, y) in enumerate(zip(optimizer._x_train, optimizer._y_train)):
        ax_post.text(x, y, str(i), color="black", fontsize=10)

    plt.show()


if __name__ == "__main__":
    main()
