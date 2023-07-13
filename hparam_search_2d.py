import numpy as np
from matplotlib import ticker
import matplotlib.pyplot as plt

from gp.kernel import RBFKernel
from gp.optimizer import GaussianProcessOptimizer


def goldstein_price_function(x1, x2=None):
    if x2 is None:
        x2 = x1[:, 1]
        x1 = x1[:, 0]

    return (
        1
        + (x1 + x2 + 1) ** 2
        * (19 - 14 * x1 + 3 * x1**2 - 14 * x2 + 6 * x1 * x2 + 3 * x2**2)
    ) * (
        30
        + (2 * x1 - 3 * x2) ** 2
        * (18 - 32 * x1 + 12 * x1**2 + 48 * x2 - 36 * x1 * x2 + 27 * x2**2)
    )


def himmelblaus_function(x1, x2=None):
    if x2 is None:
        x2 = x1[:, 1]
        x1 = x1[:, 0]

    return (x1**2 + x2 - 11) ** 2 + (x1 + x2**2 - 7) ** 2


def main():
    optimizer = GaussianProcessOptimizer(
        kernel=RBFKernel(signal_length_factor=3),
        lower_bound=[-7, -7],
        upper_bound=[7, 7],
        std_noise=0.0,
        minimize=True,
    )
    opt_x, opt_y = optimizer.optimize(himmelblaus_function, num_iterations=20)

    print(f"Optimal point: {opt_x = }, {opt_y = }")

    xs = np.linspace(-8, 8, 100)
    xx, yy = np.meshgrid(xs, xs)
    x_test = np.array((xx, yy)).T.reshape(-1, 2)
    y_true = himmelblaus_function(xx, yy)
    acq_test = optimizer._acquisition(x_test).reshape(xx.shape[0], yy.shape[0])
    mu_post, std_post, cov_post = optimizer.regressor.predict(x_test)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
    ax_acq, ax_exp, ax_post, ax_post_std = axes.reshape(-1)
    locator = ticker.LogLocator()

    # Contours
    contour_true = ax_exp.contourf(xx, yy, y_true, locator=locator)
    contour_post = ax_post.contourf(
        xx, yy, mu_post.reshape(xx.shape[0], yy.shape[0]), locator=locator
    )
    ax_post_std.contourf(xx, yy, std_post.reshape(xx.shape[0], yy.shape[0]))
    ax_acq.contourf(xx, yy, acq_test)

    # Exploration
    ax_exp.scatter(optimizer._x_train[:, 0], optimizer._x_train[:, 1], c="black", s=10)
    for i, x in enumerate(optimizer._x_train):
        ax_exp.text(x[0] - 0.12, x[1] + 0.15, str(i), color="black", fontsize=8)

    # Calculate the common minimum and maximum values for the colorbar
    vmin = min(contour_true.get_array().min(), contour_post.get_array().min())
    vmax = max(contour_true.get_array().max(), contour_post.get_array().max())
    contour_true.set_clim(vmin, vmax)
    contour_post.set_clim(vmin, vmax)

    # Plot metadata
    titles = [
        "Acquisition landscape",
        "True function",
        "Posterior $\mu$",
        "Posterior $\sigma$",
    ]
    [ax.set_title(title) for ax, title in zip(axes.reshape(-1), titles)]
    [ax.set_ylabel("$x_2$") for ax in axes[:, 0]]
    [ax.set_xlabel("$x_1$") for ax in axes[-1, :]]

    plt.show()


if __name__ == "__main__":
    main()
