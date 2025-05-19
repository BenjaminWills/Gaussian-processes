import sys

sys.path.insert(0, "..")

# Import the gaussian process claas
from gaussian_processes import One_d_gaussian_process
from kernels import rbf_kernel

# Load in the diabetes dataset
from sklearn import datasets

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
diabetes_df = datasets.load_diabetes(as_frame=True)

print(
    f"The Diabetes data has {diabetes_X.shape[1]} features, the feature names are {diabetes_df.feature_names}"
)

# Plot all features against the diabetes target variable in a subplot of 2x5
fig, axs = plt.subplots(2, 5, figsize=(20, 10))
for i in range(10):
    axs[i // 5, i % 5].scatter(diabetes_X[:, i], diabetes_y, label="Data Points")
    axs[i // 5, i % 5].set_title(diabetes_df.feature_names[i])
    axs[i // 5, i % 5].set_xlabel("Feature Value")
    axs[i // 5, i % 5].set_ylabel("Diabetes Progression")
    axs[i // 5, i % 5].legend()

# Make the plot tight, this means that the subplots will not overlap
plt.tight_layout()

plt.show()

# Is there covariance between the dependent variables in this dataset?
# The covariance matrix is a matrix that shows the covariance between each pair of features
covariance = np.cov(diabetes_X.T)

# Minmax normalise the covariance (between 0 and 1)
covariance = (covariance - np.min(covariance)) / (
    np.max(covariance) - np.min(covariance)
)

# Create a heatmap of the covariances between the independent variables
fig, ax = plt.subplots(figsize=(10, 10))
# Define the color map
cax = ax.matshow(covariance, cmap="coolwarm")
plt.colorbar(cax)

# Set the x ticks and y ticks
ax.set_xticks(np.arange(len(diabetes_df.feature_names)))
ax.set_yticks(np.arange(len(diabetes_df.feature_names)))
ax.set_xticklabels(diabetes_df.feature_names, rotation=45)
ax.set_yticklabels(diabetes_df.feature_names)

# Title the plot
plt.title("Covariance Matrix for the diabetes dataset.")
plt.show()

# We see that there is some covariance here, so the dependent variables are not independent of one another.
# Therefore we cannot create a gaussian process for each dependent variable.

# Define the training inputs
# We will use the first 50 data points as training data
training_inputs = diabetes_X[:50, :]
# Define the training outputs
training_outputs = diabetes_y[:50]

# Then the testing inputs are just the Diabetes X data.
testing_inputs = diabetes_X

# We will create a gaussian process for the first dependent variable, and then we will use that to predict the independent variable.
# Create a gaussian process for the first dependent variable

gaussian_process = One_d_gaussian_process(
    training_inputs=training_inputs,
    training_outputs=training_outputs,
    testing_range=testing_inputs,
    kernel=rbf_kernel,
    noise=0.05,
)

gaussian_process.visualise()
