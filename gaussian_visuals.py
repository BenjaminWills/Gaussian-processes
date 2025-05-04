from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt


def one_dimensional_gaussian(x: float, mean: float, variance: float) -> float:
    return (
        1 / np.sqrt(2 * np.pi * variance) * np.exp(-0.5 * ((x - mean) ** 2) / variance)
    )


if __name__ == "__main__":
    # Define parameters
    mean = 0
    variance = 1

    # Define a space of numbers
    x = np.linspace(-5, 5, 100)

    # Calculate the Gaussian function
    y = one_dimensional_gaussian(x, mean, variance)

    # Plot the Gaussian function
    plt.plot(x, y)
    plt.title("1D Gaussian Distribution")
    plt.xlabel("x")
    plt.ylabel("Probability Density")

    # Draw the mean as a vertical line
    plt.vlines(
        mean,
        ymin=0,
        ymax=one_dimensional_gaussian(mean, mean, variance),
        color="r",
        linestyle="--",
        label="Mean",
    )

    # Draw +/- 1 standard deviation
    plt.vlines(
        mean + np.sqrt(variance),
        ymin=0,
        ymax=one_dimensional_gaussian(mean + np.sqrt(variance), mean, variance),
        color="g",
        linestyle="--",
        label="+1 Std Dev",
    )
    plt.vlines(
        mean - np.sqrt(variance),
        ymin=0,
        ymax=one_dimensional_gaussian(mean - np.sqrt(variance), mean, variance),
        color="g",
        linestyle="--",
        label="-1 Std Dev",
    )

    # I want to shade the area under the curve between -1 and 1 standard deviation
    x_fill = np.linspace(mean - np.sqrt(variance), mean + np.sqrt(variance), 100)
    y_fill = one_dimensional_gaussian(x_fill, mean, variance)
    plt.fill_between(x_fill, y_fill, alpha=0.5, color="g")

    plt.legend()
    plt.grid()

    # Save the figure
    plt.savefig("1d_gaussian.png", dpi=300)

    # Show the plot
    plt.show()
