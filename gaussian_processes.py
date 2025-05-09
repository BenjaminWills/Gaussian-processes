import numpy as np
import matplotlib.pyplot as plt

from kernels import (
    rbf_kernel,
    linear_kernel,
    periodic_kernel,
    expoonentiated_quadratic_kernel,
)


def calculate_covariance_matrix(x: np.array, y: np.array, kernel: callable) -> np.array:
    """Calculates a co-variance matrix using the given kernel function.

    Parameters
    ----------
    x : np.array
    y : np.array
    kernel : callable
        A function that takes two arguments and returns a float, classic examples of these are
        the RBF kernel, the linear kernel and the periodic kernel.

    Returns
    -------
    np.array
        An array with dimensions |x| by |y|.
    """
    n = len(x)
    m = len(y)
    covariance_matrix = np.zeros((n, m))
    for i, x_i in enumerate(x):
        for j, y_j in enumerate(y):
            covariance_matrix[i, j] = kernel(x_i, y_j)
    return covariance_matrix


class One_d_gaussian_process:
    def __init__(
        self,
        training_inputs: np.array,
        training_outputs: np.array,
        testing_range: np.array,
        kernel: callable,
        noise: float = 0.05,
    ):
        self.training_inputs = training_inputs
        self.training_outputs = training_outputs
        self.testing_range = testing_range
        self.kernel = kernel
        self.noise = noise

        # Define the matrices
        self.conditioned_mean, self.conditioned_covariance = self.condition_gaussian()

    def condition_gaussian(
        self,
    ) -> tuple:
        # Aim to work out P(X | Y)

        # Partition the covariance matrix (as in the markdown)
        Sigma_XX = calculate_covariance_matrix(
            self.testing_range, self.testing_range, self.kernel
        )
        Sigma_YY = calculate_covariance_matrix(
            self.training_inputs, self.training_inputs, self.kernel
        ) + self.noise * np.eye(len(self.training_inputs))
        Sigma_XY = calculate_covariance_matrix(
            self.testing_range, self.training_inputs, self.kernel
        )
        Sigma_YX = calculate_covariance_matrix(
            self.training_inputs, self.testing_range, self.kernel
        )

        # Conditioned mean and covariance, we assume that the mean is zero
        conditioned_mean = Sigma_XY @ np.linalg.inv(Sigma_YY) @ self.training_outputs
        conditioned_covariance = (
            Sigma_XX - Sigma_XY @ np.linalg.inv(Sigma_YY) @ Sigma_YX
        )

        return conditioned_mean, conditioned_covariance

    def calculate_error_margins(self) -> tuple:
        std_dev = np.sqrt(np.diag(self.conditioned_covariance))
        return self.conditioned_mean, std_dev

    def sample_gaussian_process(self) -> np.array:
        # Define covariance matrix
        samples = np.random.multivariate_normal(
            self.conditioned_mean, self.conditioned_covariance
        )
        return samples

    def visualise(self, true_func: callable = None, save: bool = False) -> None:
        means, stds = self.calculate_error_margins()
        plt.figure(figsize=(12, 12))
        # Plot the samples
        plt.plot(
            self.testing_range,
            means,
            label=f"Gaussian process",
            color="green",
            alpha=1,
        )

        # Draw the uncertainty region around each point plotted

        plt.fill_between(
            self.testing_range,
            means - stds,
            means + stds,
            alpha=0.2,
            color="green",
            label="± 1 $\sigma$ confidence interval",
        )

        # Plot the training data
        plt.scatter(
            self.training_inputs,
            self.training_outputs,
            color="red",
            label="Training data",
        )

        # If we know the true func, plot it
        if true_func is not None:
            plt.plot(
                self.testing_range,
                [true_func(x) for x in self.testing_range],
                label="True function",
                color="blue",
                linestyle="dashed",
            )

        # Label the axes
        plt.xlabel("x")
        plt.ylabel("Samples")
        plt.title(
            "Plot of the means of the gaussian process with uncertainty margins and noisy training data"
        )
        plt.legend(loc="upper right")

        if save:
            plt.savefig("gaussian_process.png", dpi=300)

        plt.show()


if __name__ == "__main__":
    training_func = lambda x: x**2 + 3 * x + 1

    # Generate training data
    training_vals = np.linspace(-20, 20, 300)
    # Randomly sample training points from a big list
    num_training_points = 30
    training_vals = np.random.choice(training_vals, num_training_points, replace=False)
    training_outputs = np.array([training_func(x) for x in training_vals])

    # Generate testing values
    testing_vals = np.linspace(-20, 20, 500)

    # Create the Gaussian process
    gp = One_d_gaussian_process(
        training_inputs=training_vals,
        training_outputs=training_outputs,
        testing_range=testing_vals,
        kernel=rbf_kernel,
        noise=1e-50,
    )
    # Plot the gaussian process
    gp.visualise(
        true_func=training_func,
    )
