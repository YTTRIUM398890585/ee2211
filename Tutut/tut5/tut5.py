
import numpy as np

import matplotlib.pyplot as plt


import pandas as pd

from sklearn.metrics import mean_squared_error

import sys
sys.path.append('../../')   # nopep8
import ee2211pythonhelper as helper  # nopep8


# Question 1 ########################################################################################
print("Question 1 ########################################################################################")

X = np.array([[-10], [-8], [-3], [-1], [2], [8]])
Y = np.array([[5], [5], [4], [3], [2], [2]])

W_bias = helper.linearRegressionWithBias(
    X, Y, printResult=True, printFeature=1)

W_nobias = helper.linearRegressionWithoutBias(
    X, Y, printResult=True, printFeature=0)

# Question 2 ########################################################################################
print("Question 2 ########################################################################################")

X = np.array([[1, 0, 1], [2, -1, 1], [1, 1, 5]])
Y = np.array([[1], [2], [3]])

Xtest1 = np.array([[-1, 2, 8]])
Xtest2 = np.array([[1, 5, -1]])

W_nobias = helper.linearRegressionWithoutBias(
    X, Y, printResult=True, printFeature=None)

Ytest1_nobias = helper.testData(Xtest1, W_nobias, printResult=False)
print("Ytest1_nobias =\n", Ytest1_nobias)
Ytest2_nobias = helper.testData(Xtest2, W_nobias, printResult=False)
print("Ytest2_nobias =\n", Ytest2_nobias)

# this is wrong, should use right inverse
W_bias = helper.linearRegressionWithBias(
    X, Y, printResult=True, printFeature=None)

# add in bias
Xtest1 = np.array([[1, -1, 2, 8]])
Xtest2 = np.array([[1, 1, 5, -1]])

Ytest1_bias = helper.testData(Xtest1, W_bias, printResult=False)
print("Ytest1_bias =\n", Ytest1_bias)
Ytest2_bias = helper.testData(Xtest2, W_bias, printResult=False)
print("Ytest2_bias =\n", Ytest2_bias)

# after adding bias, its a wider matrix so should solve with right inverse
W_bias_right = helper.rightInverse(helper.paddingOfOnes(X)) @ Y
print("W_bias_right =\n", W_bias_right)

Ytest1_bias = helper.testData(Xtest1, W_bias_right, printResult=False)
print("Ytest1_bias =\n", Ytest1_bias)
Ytest2_bias = helper.testData(Xtest2, W_bias_right, printResult=False)
print("Ytest2_bias =\n", Ytest2_bias)


# Question 3 ########################################################################################
print("Question 3 ########################################################################################")

X = np.array([[36], [28], [35], [39], [30], [30],
             [31], [38], [36], [38], [29], [26]])
Y = np.array([[31], [29], [34], [35], [29], [30],
             [30], [38], [34], [33], [29], [26]])

plt.plot(X, Y, 'x')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# xw = Y regression
w = helper.linearRegressionWithBias(X, Y, printResult=True, printFeature=1)

# predict
Xtest1 = np.array([[1, 30]])
Xtest2 = np.array([[1, 5]])

Ytest1 = helper.testData(Xtest1, w, printResult=False)
print("Ytest1 =\n", Ytest1)
Ytest2 = helper.testData(Xtest2, w, printResult=False)
print("Ytest2 =\n", Ytest2)

# Question 4 ########################################################################################
print("Question 4 ########################################################################################")

X = np.array([[36], [26], [35], [39], [26], [30],
             [31], [38], [36], [38], [26], [26]])
Y = np.array([[31], [20], [34], [35], [20], [30],
             [30], [38], [34], [33], [20], [20]])

# xw = Y regression
w = helper.linearRegressionWithBias(X, Y, printResult=True, printFeature=1)

# predict
Xtest1 = np.array([[1, 30]])
Ytest1 = helper.testData(Xtest1, w, printResult=False)
print("Ytest1 =\n", Ytest1)

# remove duplicates
print("X =\n", X)
print("Y =\n", Y)
combined = np.hstack((X, Y))
print("combined =\n", combined)
new_array = [tuple(row) for row in combined]
print("new_array =\n", new_array)
uniques = np.unique(new_array, axis=0)
print("uniques =\n", uniques)
X_unique = uniques[:, 0].reshape(-1, 1)
Y_unique = uniques[:, 1].reshape(-1, 1)
# In this context, -1 is used as a placeholder for "figure out what the dimension should be". It means that the length in that dimension is inferred from the length of the array and remaining dimensions.
# So, .reshape(-1, 1) means to reshape the array to have one column and as many rows as necessary to accommodate the original data.
print("X_unique =\n", X_unique)
print("Y_unique =\n", Y_unique)

# xw = Y regression
w_unique = helper.linearRegressionWithBias(
    X_unique, Y_unique, printResult=True, printFeature=1)

# predict
Ytest1_unique = helper.testData(Xtest1, w_unique, printResult=False)
print("Ytest1_unique =\n", Ytest1_unique)

# Combining the plots for easier comparison
plt.plot(X, Y, 'o', color='blue')

# plot the graph for the unique data, X needs to be padded for the offset
plt.plot(X, np.hstack((np.ones((X.shape[0], 1)), X)) @ w, '-', color='red')

# plot the graph for the unique data
plt.plot(X, np.hstack(
    (np.ones((X.shape[0], 1)), X)) @ w_unique, '-', color='green')

plt.xlabel('X')
plt.ylabel('Y')

plt.show()

# Question 5 ########################################################################################
print("Question 5 ########################################################################################")
exp_df = pd.read_csv("GovernmentExpenditureonEducation.csv")
exp_df.info()

X = np.array(exp_df['year']).reshape(-1, 1)
Y = np.array(exp_df['total_expenditure_on_education']).reshape(-1, 1)
print("X =\n", X)
print("Y =\n", Y)

# xw = Y regression
W = helper.linearRegressionWithBias(X, Y, printResult=True, printFeature=1)

x_test = helper.paddingOfOnes(np.array([[2021]]))

y_predict = helper.testData(x_test, W, printResult=False)

print("y_predict =\n", y_predict)

# Question 6 ########################################################################################
print("Question 6 ########################################################################################")

# wine_df = pd.read_csv("wine.csv", sep=';')
# print(wine_df)

wine_df = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=";")
wine_df.info()

# drop the quality column, the other columns are the features
X = np.array(wine_df.drop('quality', axis=1))
# the quality column is the output
Y = np.array(wine_df['quality'].tolist()).reshape(-1, 1)

print("X\n", X)
print("Y\n", Y)

Xtrain = X[:1500]
Ytrain = Y[:1500]
print("Xtrain\n", Xtrain)
print("Ytrain\n", Ytrain)

Xtest = X[1500:]
Ytest = Y[1500:]
print("Xtest\n", Xtest)
print("Ytest\n", Ytest)

# Training
w = helper.linearRegressionWithBias(
    Xtrain, Ytrain, printResult=True, printFeature=None)

# Prediction Test
Ytest_pred = helper.testData(helper.paddingOfOnes(Xtest), w, printResult=False)
print("Ytest_pred\n", Ytest_pred)

MSE = mean_squared_error(Ytest_pred, Ytest)
print("MSE = ", MSE)

# Question 7 ########################################################################################
print("Question 7 ########################################################################################")

X = np.array([[3, -1, 0], [5, 1, 2], [9, -1, 3], [-6, 7, 2], [3, -2, 0]])
Y = np.array([[1, -1], [-1, 0], [1, 2], [0, 3], [1, -2]])

# Training
w = helper.linearRegressionWithBias(X, Y, printResult=True, printFeature=None)

Xtest = helper.paddingOfOnes(np.array([[8, 0, 2]]))
Ytest = helper.testData(Xtest, w, printResult=False)
print("Ytest\n", Ytest)
