import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from tqdm import tqdm

from gaussian_processes import One_d_gaussian_process
from acquisition_functions import (
    probability_of_improvement,
    expected_improvement,
)
from kernels import rbf_kernel


class One_d_gaussian_process_optimiser:
    def __init__(
        self,
        evaluation_function: callable,
        lower_bound: float,
        upper_bound: float,
        kernel: callable,
        aqcuisition_function: callable,
        epsilon: float = 0.3,
    ):
        """This class is a one dimensional gaussian proess optimiser. It is used to
        optimise a function in one dimension. It uses a gaussian process to model the
        function, and an acquisition function to find the next point to sample and then
        to update the gaussian process.

        Parameters
        ----------
        evaluation_function : callable
            The function that we want to optimise. It should take a single input and
            return a single output. Note that the functions usually optimised in this
            way are expensive to evaluate, so we want to sample as few points as possible.
        lower_bound : float
            The lower bound of the search space.
        upper_bound : float
            The upper bound of the search space.
        kernel : callable
            The kernel function to use for the gaussian process. It should take two inputs
            and return a single output.
        aqcuisition_function : callable
            The acquisition function to use for the gaussian process. It should take
            three inputs: the input point, the gaussian process and the maximum value of
            the function at the given timestamp. It should return a single output.
        epsilon : float, optional
            The exploration / exploitation trade off. High epsilon means
            high amounts of exploitation in regions with high
            standard deviation, by default 0.3
        """
        # Define the evaluation function to sample points from
        self.evaluation_function = evaluation_function

        # Define the intial of the search
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        # Define the kernel function
        self.kernel = kernel

        # Define the acquisition function
        self.acquisition_function = aqcuisition_function

        # Define the exploration / exploitation trade off
        self.epsilon = epsilon

    def choose_random_point(self):
        return np.random.uniform(self.lower_bound, self.upper_bound)

    def choose_next_point(
        self,
        fitted_gaussian_process: One_d_gaussian_process,
        acquisition_function: callable,
        maximum_value: float,
    ) -> float:

        # Get the next point using the probability of improvement, we want to omptimise
        # the acquisiiton function
        objective_function = lambda x: -acquisition_function(
            x,
            fitted_gaussian_process,
            maximum_value,
            epsilon=self.epsilon,
        )
        next_point = minimize_scalar(
            objective_function,
            bounds=(self.lower_bound, self.upper_bound),
            method="bounded",
        )

        return next_point.x

    def optimise(self, max_iterations: int = 10_000):
        # First we sample a random point
        x = self.choose_random_point()

        # Evaluate the function at that point
        y = self.evaluation_function(x)

        # Store the maximum value of the function
        maximum_func_value = y

        # Create the training data
        training_data = np.array([x])
        training_outputs = np.array([y])

        # Define the domain on which we will test
        testing_range = np.linspace(self.lower_bound, self.upper_bound, 1000)

        # Create the initial gaussian process (prior surrogate model), and calcualte
        # the posterior distribution given the data
        fitted_gaussian_process = One_d_gaussian_process(
            training_inputs=training_data,
            training_outputs=training_outputs,
            testing_range=testing_range,
            kernel=self.kernel,
        )

        # Now we begin the loop, the stopping criteria are as follows:
        # 1. If the max number of iterations has been reached
        # 2. If the maximum value of the function has not changed in the last 10 iterations

        # Make stagnation iteration counter
        stagnation_counter = 0

        for iteration in tqdm(range(max_iterations), desc="Iteration number: "):

            # Find the current maximum value
            maximum_func_value = np.max(training_outputs)

            # Choose a next point given the current gaussian process, based on
            next_point = self.choose_next_point(
                fitted_gaussian_process=fitted_gaussian_process,
                acquisition_function=self.acquisition_function,
                maximum_value=maximum_func_value,
            )

            # Update the training arrays
            training_data = np.append(training_data, next_point)
            function_value = self.evaluation_function(next_point)
            training_outputs = np.append(training_outputs, function_value)

            # If the next point is bigger the maximum value, update the maximum value
            if function_value > maximum_func_value:
                stagnation_counter = 0
                maximum_func_value = function_value
                print(f"New maximum value found: {maximum_func_value} at {next_point}")
            else:
                stagnation_counter += 1

            # If the maximum value has not changed in the last 10 iterations, break
            if stagnation_counter > max_iterations / 2:
                print(
                    f"Maximum value has not changed in the last 10 iterations, stopping optimisation."
                )
                break

            # Update the gaussian process with the new data
            fitted_gaussian_process = One_d_gaussian_process(
                training_inputs=training_data,
                training_outputs=training_outputs,
                testing_range=testing_range,
                kernel=self.kernel,
            )
        return fitted_gaussian_process, training_data, training_outputs


if __name__ == "__main__":
    # Define a function to be optimised
    def func(x):
        return x * np.cos(x)

    optimiser = One_d_gaussian_process_optimiser(
        evaluation_function=func,
        lower_bound=-10,
        upper_bound=10,
        kernel=rbf_kernel,
        aqcuisition_function=probability_of_improvement,
        epsilon=1,
    )

    fitted_gaussian_process, training_data, training_outputs = optimiser.optimise(
        max_iterations=100
    )

    # Plot the results
    plt.figure(figsize=(12, 12))

    # Plot the maximum value
    max_x = training_data[np.argmax(training_outputs)]
    max_y = np.max(training_outputs)
    print(f"Maximum value found: {max_y} at x = {max_x}")

    # Plot the acquisition function at the end
    plt.plot(
        fitted_gaussian_process.testing_range,
        [
            probability_of_improvement(x, fitted_gaussian_process, max_y)
            for x in fitted_gaussian_process.testing_range
        ],
        label="Acquisition function",
        color="green",
    )
    plt.xlabel("x value")
    plt.ylabel("Aquisition function value, higher is better")
    plt.title("Acquisition function")
    plt.savefig("aquisition_function.png", dpi=300)
    plt.show()

    # Plot the samples
    fitted_gaussian_process.visualise(func, save=True)
