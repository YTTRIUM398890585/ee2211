from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import pandas as pd

from pandas.plotting import scatter_matrix

import matplotlib.pyplot as plt

# import sklearn
# print("pandas version: {}".format(pd.__version__))
# print("scikit-learn version: {}".format(sklearn.__version__))

# Load the iris dataset
iris_dataset = load_iris()

# print(iris_dataset)

# Split the dataset into training and test sets, random_state is set to 0 to ensure the same output
# random_state is the seed used by the random number generator
X_train, X_test, Y_train, Y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

# print("X_train", X_train)
# print("X_test", X_test)

# print("Y_train", Y_train)
# print("Y_test", Y_test)

# Create a DataFrame from the training set
# columns are needed to label the columns, otherwise the columns will be labeled with numbers
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)

# print(iris_dataframe)

# Create a scatter matrix from the dataframe, color by Y_train
# figsize is used to set the size of the figure
# marker is set to '0' to use a circle as the marker
# hist_kwds is used to set the number of bins for the histograms
grr = scatter_matrix(iris_dataframe, c=Y_train, figsize=(15, 15), marker='0', hist_kwds={'bins': 20})

# Show the scatter matrix
plt.show()

grr = scatter_matrix(iris_dataframe)
# Show the scatter matrix
plt.show()
