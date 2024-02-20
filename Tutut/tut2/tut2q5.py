import pandas as pd
import numpy as np

dataset = pd.read_csv("pima-indians-diabetes.data.csv", header=None)
print(dataset)

# this prints the summary statistics of the dataset
print(dataset.describe())

# this prints the number of missing values in each column, separately as a series (pandas data structure)
print((dataset[[1, 2, 3, 4, 5]] == 0).sum())

# this prints the total number of missing values in the dataset
print((dataset[[1, 2, 3, 4, 5]] == 0).sum().sum())

# prints the first 20 rows of the dataset
print(dataset.head(20))

# replace the 0 values in columns 1, 2, 3, 4, 5 with NaN
dataset[[1, 2, 3, 4, 5]] = dataset[[1, 2, 3, 4, 5]].replace(0, np.NaN)

# prints the first 20 rows of the dataset
print(dataset.head(20))