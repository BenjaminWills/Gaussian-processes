from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt


def rbf_kernel(t_1: float, t_2: float, sigma: float = 3, length: float = 2) -> float:
    return sigma**2 * np.exp(-0.5 * ((t_1 - t_2) / length) ** 2)


def periodic_kernel(
    t_1: float,
    t_2: float,
    periodicity: float = 0.2,
    sigma: float = 0.75,
    length: float = 3,
) -> float:
    return sigma**2 * np.exp(
        2 * np.sin(np.pi * abs(t_1 - t_2) / periodicity) ** 2 / length**2
    )


def calculate_covariance_matrix(x: np.array, kernel: callable) -> np.array:
    n = len(x)
    covariance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            covariance_matrix[i, j] = kernel(x[i], x[j])
    return covariance_matrix


# def conditional_paramaters(
#     conditional_value: float, covariance_matrix: np.array
# ) -> tuple:
#     # Marganalised params
#     mean_marginalise = (
#         covariance_matrix[1, 1]
#         + covariance_matrix[1, 0]
#         * (conditional_value - covariance_matrix[0, 0])
#         / covariance_matrix[1, 1]
#     )
#     var_marginalise = (
#         covariance_matrix[1, 1]
#         - covariance_matrix[1, 0] * covariance_matrix[0, 1] / covariance_matrix[0, 0]
#     )
#     return mean_marginalise, var_marginalise


def sample_gaussian_process(
    x_vals: np.array, kernel: callable = rbf_kernel
) -> np.array:

    # Define covariance matrix
    covariance_matrix = calculate_covariance_matrix(x_vals, kernel)

    # Calculate marginalised parameters for each value of x
    # samples = []
    # for x_marginalise in x_vals:
    #     mean_marginalise, var_marginalise = conditional_paramaters(
    #         x_marginalise, covariance_matrix
    #     )
    #     # Sample from a normal distribution with these parameters
    #     sample = np.random.normal(mean_marginalise, np.sqrt(var_marginalise))
    #     samples.append(sample)

    # Sample from a normal distribution with these parameters
    mean = np.zeros(len(x_vals))
    samples = np.random.multivariate_normal(mean, covariance_matrix)

    return samples


if __name__ == "__main__":

    x_vals = np.linspace(-10, 10, 100)

    for i in range(4):
        # Generate samples
        samples = sample_gaussian_process(x_vals, kernel=rbf_kernel)
        # Plot the samples
        plt.plot(x_vals, samples, label=f"Sample {i+1}", alpha=0.7)
    plt.xlabel("x")
    plt.ylabel("Samples")
    plt.title("Samples from Gaussian with RBF covariance matrix")
    plt.legend()
    plt.savefig("rbf_kernel_samples_.png", dpi=300)
    plt.show()
