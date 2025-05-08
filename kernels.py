import numpy as np


def rbf_kernel(
    x1: float, x2: float, length_scale: float = 2, sigma: float = 3
) -> float:
    """
    Radial Basis Function (RBF) kernel, also known as the Gaussian kernel.

    Parameters:
        x1 (float): First input scalar.
        x2 (float): Second input scalar.
        length_scale (float): Length scale parameter (must be positive).
        sigma (float): Variance parameter (default is 3).

    Returns:
        float: The computed RBF kernel value.

    Raises:
        ValueError: If length_scale is not positive.
    """
    if length_scale <= 0:
        raise ValueError("length_scale must be positive.")
    distance = abs(x1 - x2)
    return sigma**2 * np.exp(-0.5 * (distance / length_scale) ** 2)


def linear_kernel(
    x1: float, x2: float, c: float = 0.0, variance: float = 0.2, variance_b: float = 0.2
) -> float:
    """
    Linear kernel.

    Parameters:
        x1 (float): First input scalar.
        x2 (float): Second input scalar.
        c (float): Bias term (default is 0.0).
        variance (float): Variance parameter (default is 0.2).
        variance_b (float): Variance parameter surrounding C (default is 0.2).

    Returns:
        float: The computed Linear kernel value.
    """
    return variance_b**2 + variance**2 * (x1 - c) * (x2 - c)


def periodic_kernel(
    x1: float, x2: float, length_scale: float, period: float, variance: float
) -> float:
    """
    Periodic kernel.

    Parameters:
        x1 (float): First input scalar.
        x2 (float): Second input scalar.
        length_scale (float): Length scale parameter (must be positive).
        period (float): Period parameter (must be positive).
        variance (float): Variance parameter.

    Returns:
        float: The computed Periodic kernel value.

    Raises:
        ValueError: If length_scale or period is not positive.
    """
    if length_scale <= 0:
        raise ValueError("length_scale must be positive.")
    if period <= 0:
        raise ValueError("period must be positive.")
    distance = abs(x1 - x2)
    return variance**2 * np.exp(
        -2 * (np.sin(np.pi * distance / period) ** 2) / length_scale**2
    )


def expoonentiated_quadratic_kernel(
    x1: float, x2: float, variance: float, length: float
) -> float:
    """
    Quadratic kernel.

    Parameters:
        x1 (float): First input scalar.
        x2 (float): Second input scalar.
        variance (float): Variance parameter.
        length (float): Length scale parameter (must be positive).

    Returns:
        float: The computed Quadratic kernel value.
    """

    if length <= 0:
        raise ValueError("length must be positive.")

    return variance**2 * np.exp(
        -0.5 * ((x1 - x2) / length) ** 2
    )  # Exponential of the negative squared distance divided by length scale
