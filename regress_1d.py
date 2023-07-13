import numpy as np
import matplotlib.pyplot as plt

from gp.kernel import RBFKernel
from gp.regressor import GaussianProcessRegressor


def true_function(x: np.ndarray) -> float:
    return np.sin(2 * x[:, 0] - 2) * np.cos(0.5 * x[:, 0] + 3)


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


def main():
    domain = (-5, 5)
    n_samples = 85
    n_functions = 5

    x_prior = np.linspace(*domain, n_samples, dtype=np.float32)[:, None]
    cov_prior = RBFKernel()(x_prior, x_prior)
    std_prior = np.sqrt(np.diag(cov_prior))

    mu_prior = np.zeros(n_samples, dtype=np.float32)
    ys_prior = np.random.multivariate_normal(
        mean=mu_prior, cov=cov_prior, size=n_functions
    )

    n_training = 10  # num points to condition on
    n_test = 75  # num points to posterio
    n_post_functions = 5  # num functions to sample from posterior

    # Sample (X1, y1)
    x_train = np.random.uniform(domain[0] + 1, domain[1] - 1, size=(n_training, 1))
    y_train = true_function(x_train)

    # Predict uniformly to define the function (we would be smarter in gaussian optimization later)
    x_test = np.linspace(*domain, n_test)[:, None]

    # Get posterior mean and cov
    gp = GaussianProcessRegressor(RBFKernel(), std_noise=0.1).fit(x_train, y_train)

    mu_post, std_post, cov_post = gp.predict(x_test)
    ys_test = np.random.multivariate_normal(
        mean=mu_post, cov=cov_post, size=n_post_functions
    )

    _, (ax_prior, ax_post) = plt.subplots(1, 2, figsize=(16, 4))
    plot_landscape(
        None,
        None,
        x_prior,
        ys_prior,
        mu_prior,
        std_prior,
        ax_prior,
        title="prior",
        xlim=domain,
    )
    plot_landscape(
        x_train,
        y_train,
        x_test,
        ys_test,
        mu_post,
        std_post,
        ax_post,
        title="posterior",
        xlim=domain,
    )
    plt.show()


if __name__ == "__main__":
    main()
