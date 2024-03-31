import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd

import sys
sys.path.append('../../')   # nopep8
import ee2211pythonhelper as helper  # nopep8

from sklearn.datasets import fetch_california_housing

''' Question 3 '''
housing = fetch_california_housing()
print(housing)


# Convert to DataFrame
data_df = pd.DataFrame(housing['data'], columns=housing['feature_names'])
data_df = pd.DataFrame(housing['target'])

# Select features
Xtrain = data_df["MedInc"]

print(Xtrain)


''' Question 4 '''
