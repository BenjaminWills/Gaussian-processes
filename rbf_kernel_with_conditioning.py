from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt


def rbf_kernel(t_1: float, t_2: float, sigma: float = 3, length: float = 2) -> float:
    return sigma**2 * np.exp(-0.5 * ((t_1 - t_2) / length) ** 2)


def calculate_covariance_matrix(x: np.array, y: np.array, kernel: callable) -> np.array:
    n = len(x)
    m = len(y)
    covariance_matrix = np.zeros((n, m))
    for i, x_i in enumerate(x):
        for j, y_j in enumerate(y):
            covariance_matrix[i, j] = kernel(x_i, y_j)
    return covariance_matrix


def condition_gaussian(
    training_data: np.array,
    testing_data: np.array,
    training_outputs: np.array,
    kernel: callable,
) -> tuple:
    # Aim to work out P(X | Y)

    # Partition the covariance matrix (as in the markdown)
    Sigma_XX = calculate_covariance_matrix(testing_data, testing_data, kernel)
    Sigma_YY = calculate_covariance_matrix(training_data, training_data, kernel)
    Sigma_XY = calculate_covariance_matrix(testing_data, training_data, kernel)
    Sigma_YX = calculate_covariance_matrix(training_data, testing_data, kernel)

    # Conditioned mean and covariance, we assume that the mean is zero
    conditioned_mean = Sigma_XY @ np.linalg.inv(Sigma_YY) @ training_outputs
    conditioned_covariance = Sigma_XX - Sigma_XY @ np.linalg.inv(Sigma_YY) @ Sigma_YX

    return conditioned_mean, conditioned_covariance


def sample_conditional_gaussian_process(
    training_data: np.array,
    testing_data: np.array,
    training_outputs: np.array,
    kernel: callable = rbf_kernel,
) -> np.array:
    # Define covariance matrix
    mean, covariance_matrix = condition_gaussian(
        training_data, testing_data, training_outputs, kernel
    )
    samples = np.random.multivariate_normal(mean, covariance_matrix)
    return samples


if __name__ == "__main__":

    training_func = lambda x: np.cos(x)
    num_training_points = 4
    # Generate training data
    training_vals = np.linspace(-5, 5, num_training_points)
    training_outputs = np.array([training_func(x) for x in training_vals])
    testing_vals = np.linspace(-5, 5, 100)

    for i in range(4):
        # Generate samples
        samples = sample_conditional_gaussian_process(
            training_data=training_vals,
            training_outputs=training_outputs,
            testing_data=testing_vals,
            kernel=rbf_kernel,
        )
        # Plot the samples
        plt.plot(
            testing_vals,
            samples,
            label=f"Sample {i+1}",
            alpha=0.7,
        )
    plt.plot(
        testing_vals,
        training_func(testing_vals),
        color="black",
        label="True function",
        linestyle="dashed",
    )
    plt.scatter(training_vals, training_outputs, color="red", label="Training data")
    plt.xlabel("x")
    plt.ylabel("Samples")
    plt.title("Samples from Gaussian with RBF covariance matrix")
    plt.legend()
    plt.savefig("rbf_conditional_kernel_samples_.png", dpi=300)
    plt.show()
