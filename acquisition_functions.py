import numpy as np
from scipy.stats import norm

from gaussian_processes import One_d_gaussian_process


def probability_of_improvement(
    x: float,
    gaussian_process: One_d_gaussian_process,
    maximium_function_value: float,
    epislon: float = 0.3,
) -> float:
    """This function is used to calculate the probability of improvement for
    a gaussian process. It can be used to find optimal points.

    Parameters
    ----------
    x : float
        The input point that we want to find the probability of improvement for.
    gaussian_process : One_d_gaussian_process
        The gaussian process at the given timestamp.
    maximium_function_value : float
        The maximum value of the function at the given timestamp.
    epislon : float, optional
        The exploration / exploitation trade off. High epsilon means
        high amounts of exploitation in regions with high
        standard deviation, by default 0.3

    Returns
    -------
    float
        The probability that the given x-co-ordinate will lead to an improvement of
        the function
    """
    # Find the index thats close to the input x within the testing data
    index = np.argmin(np.abs(gaussian_process.testing_range - x))
    # Get the mean and covariance of the gaussian process
    mean = gaussian_process.conditioned_mean[index]
    variance = gaussian_process.conditioned_covariance[index, index]

    # Normalise the probability
    z = (mean - maximium_function_value - epislon) / np.sqrt(variance)

    # Return the probability of improvement
    return norm.cdf(z)


def expected_improvement(
    x: float,
    gaussian_process: One_d_gaussian_process,
    maximium_function_value: float,
    epsilon: float = 0.3,
) -> float:
    """This function is used to calculate the expected improvement for
    a gaussian process. It can be used to find optimal points.

    Parameters
    ----------
    x : float
        The input point that we want to find the expected improvement for.
    gaussian_process : One_d_gaussian_process
        The gaussian process at the given timestamp.
    maximium_function_value : float
        The maximum value of the function at the given timestamp.
    epislon : float, optional
        The exploration / exploitation trade off. High epsilon means
        high amounts of exploitation in regions with high
        standard deviation, by default 0.3

    Returns
    -------
    float
        The expected improvement of the given x-co-ordinate
        in the function.
    """
    # Find the index thats close to the input x within the testing data
    index = np.argmin(np.abs(gaussian_process.testing_range - x))
    # Get the mean and covariance of the gaussian process
    mean = gaussian_process.conditioned_mean[index]
    variance = gaussian_process.conditioned_covariance[index, index]

    if variance == 0:
        return 0

    # Normalise the probability
    z = (mean - maximium_function_value - epsilon) / np.sqrt(variance)

    # Calculate the expected improvement
    expected_improvement = (mean - maximium_function_value - epsilon) * norm.cdf(
        z
    ) + np.sqrt(variance) * norm.pdf(z)
    return expected_improvement
