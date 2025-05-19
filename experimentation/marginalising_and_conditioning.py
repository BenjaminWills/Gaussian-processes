from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt


def two_dimensional_gaussian(
    x: np.array, mean: np.array, covariance_matrix: np.array
) -> float:
    covar_determinant = np.linalg.det(covariance_matrix)
    covar_inverse = np.linalg.inv(covariance_matrix)
    return (1 / (2 * np.pi * np.sqrt(covar_determinant))) * np.exp(
        -0.5 * ((x - mean).T @ covar_inverse @ (x - mean))
    )


def one_dimensional_gaussian(x: float, mean: float, variance: float) -> float:
    return (
        1 / np.sqrt(2 * np.pi * variance) * np.exp(-0.5 * ((x - mean) ** 2) / variance)
    )


if __name__ == "__main__":
    # Define parameters
    mean = np.array([0, 0])
    covariance_matrix = np.array([[1, 0.3], [0.8, 1]])
    # ( XX,XY )
    # ( YX, YY)

    # Choose the value of X to marginalise over
    x_marginalise = 0.2

    # Marganalised params
    mean_marginalise = (
        covariance_matrix[1, 1]
        + covariance_matrix[1, 0]
        * (x_marginalise - covariance_matrix[0, 0])
        / covariance_matrix[1, 1]
    )
    var_marginalise = (
        covariance_matrix[1, 1]
        - covariance_matrix[1, 0] * covariance_matrix[0, 1] / covariance_matrix[0, 0]
    )

    # Define a space of numbers
    x1 = np.linspace(-5, 5, 100)
    x2 = np.linspace(-5, 5, 100)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.zeros(X1.shape)

    # Calculate the Gaussian function
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            x = np.array([X1[i, j], X2[i, j]])
            Z[i, j] = two_dimensional_gaussian(x, mean, covariance_matrix)

    # Plot the Gaussian function
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Draw the slice of the Gaussian function at x_marginalise
    x_slice = np.linspace(-5, 5, 100)
    y_slice = np.zeros(x_slice.shape)
    for i in range(x_slice.shape[0]):
        x = np.array([x_marginalise, x_slice[i]])
        y_slice[i] = two_dimensional_gaussian(x, mean, covariance_matrix)
    ax.plot(
        x_marginalise * np.ones(x_slice.shape),
        x_slice,
        y_slice,
        color="r",
        linestyle="--",
        label="Marginalised Slice",
        zorder=1,
    )

    x_slice = np.linspace(-5, 5, 100)
    y_slice = np.zeros(x_slice.shape)
    for i in range(x_slice.shape[0]):
        x = x_slice[i]
        y_slice[i] = 0.39 * one_dimensional_gaussian(
            x, mean_marginalise, var_marginalise
        )
    ax.plot(
        x_marginalise * np.ones(x_slice.shape),
        x_slice,
        y_slice,
        color="g",
        linestyle="--",
        label="Slice according to params",
    )

    # Add the normal plot
    # ax.plot_surface(X1, X2, Z, cmap="viridis", edgecolor="none", zorder=2, alpha=0.6)
    ax.set_title("2D Gaussian Distribution")
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("Probability Density")
    plt.legend()

    # Show the figure
    plt.show()
