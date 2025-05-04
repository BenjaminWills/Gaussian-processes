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


if __name__ == "__main__":
    # Define paramaters
    mean = np.array([0, 0])
    covariance_matrix = np.array([[1, 0], [0, 1]])

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
    ax.plot_surface(X1, X2, Z, cmap="viridis", edgecolor="none")
    ax.set_title("2D Gaussian Distribution")
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("Probability Density")
    plt.savefig("2d_gaussian.png", dpi=300)
    plt.show()
