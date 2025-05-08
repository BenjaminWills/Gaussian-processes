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
    psi: float,
) -> tuple:
    # Aim to work out P(X | Y)

    # Partition the covariance matrix (as in the markdown)
    Sigma_XX = calculate_covariance_matrix(testing_data, testing_data, kernel)
    Sigma_YY = calculate_covariance_matrix(
        training_data, training_data, kernel
    ) + psi * np.eye(len(training_data))
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
    psi: float = 0.1,
) -> np.array:
    # Define covariance matrix
    mean, covariance_matrix = condition_gaussian(
        training_data, testing_data, training_outputs, kernel, psi
    )
    samples = np.random.multivariate_normal(mean, covariance_matrix)
    return samples


def calculate_error_margins(
    training_data: np.array,
    testing_data: np.array,
    training_outputs: np.array,
    kernel: callable,
    psi: float = 0.1,
) -> tuple:
    mean, covariance_matrix = condition_gaussian(
        training_data, testing_data, training_outputs, kernel, psi
    )
    std_dev = np.sqrt(np.diag(covariance_matrix))
    return mean, std_dev


if __name__ == "__main__":

    training_func = lambda x: np.cos(x)
    num_training_points = 6
    # Generate training data
    training_vals = np.linspace(-5, 5, num_training_points)
    training_outputs = np.array([training_func(x) for x in training_vals])
    testing_vals = np.linspace(-10, 10, 100)

    # Generate samples
    means, stds = calculate_error_margins(
        training_data=training_vals,
        training_outputs=training_outputs,
        testing_data=testing_vals,
        kernel=rbf_kernel,
        psi=0.5,
    )
    plt.figure(figsize=(12, 12))
    # Plot the samples
    plt.plot(
        testing_vals,
        means,
        label=f"Gaussian process",
        alpha=1,
    )
    # Draw the uncertainty region around each point plotted
    plt.fill_between(
        testing_vals,
        means - stds,
        means + stds,
        alpha=0.2,
        color="green",
        label="±1 std confidence interval",
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
    plt.title(
        "Plot of the means of the gaussian process with uncertainty margins and noisy training data"
    )
    plt.legend(loc="upper right")
    plt.savefig(
        "rbf_conditional_kernel_samples_with_uncertainty margins_and_noisy_training_data.png",
        dpi=300,
    )
    plt.show()
