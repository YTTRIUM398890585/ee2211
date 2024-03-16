import sys
sys.path.append('../../')   # nopep8
import ee2211pythonhelper as helper  # nopep8

import numpy as np

import matplotlib.pyplot as plt

from numpy.linalg import inv

import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

# Example
'''
X = np.array([[1, 2], [0, 6], [1, 0], [0, 5], [1, 7]])
Y = np.array([[1], [2], [3], [4], [5]])

w = helper.linearRegressionWithBias(X, Y, printResult=True, printFeature=None)

xtest = np.array([[1, 3]])
xtest = helper.paddingOfOnes(xtest)
ytest = helper.testData(xtest, w, True)
'''

# X = np.array([[5, 0], [6, -6], [8, 11], [4, 1]])
# a = helper.rightInverse(X)
# print(a)
# a = helper.leftInverse(X)
# print(a)

X = np.array([[45, 9], [50, 10], [63, 12], [70, 8], [80, 4]])
Y1 = np.array([[6], [9], [8], [3], [2]])
w1 = helper.linearRegressionWithBias(X, Y1, printResult=True, printFeature=None)

Y2 = np.array([[5], [6], [9], [2], [4]])
w2 = helper.linearRegressionWithBias(X, Y2, printResult=True, printFeature=None)

X_test = np.array([[63, 9]])
X_test = helper.paddingOfOnes(X_test)
ytest1 = helper.testData(X_test, w1, True)
ytest2 = helper.testData(X_test, w2, True)

# X = np.array([[4, -3, 6], [1, 0, 10]])
# a = helper.rightInverse(X)
# print(a)
# print(X @ a)
# a = helper.leftInverse(X)
# print(a)
# print(a @ X)

# X = np.array([[2, 5, 6, 7], [-1, 3, -3, 8], [7, 9, 8, 2]])
# Y1 = np.array([[3], [5], [6]])
# w1 = helper.linearRegressionWithBias(X, Y1, printResult=True, printFeature=None)

X = np.array([[5, -6], [2, 0], [4, 7], [11, -8]])
a = helper.leftInverse(X)
print(a)
Y1 = np.array([[3], [-5.5], [9], [1]])
w1 = helper.linearRegressionWithBias(X, Y1, printResult=True, printFeature=None)





