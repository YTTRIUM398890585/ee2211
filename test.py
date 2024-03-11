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

X = np.array([[1, 2], [0, 6], [1, 0]])
leftinverseX = helper.leftInverse(X)
print("leftinverseX =\n ", leftinverseX)
rightinverseX = helper.rightInverse(X)
print("rightinverseX =\n", rightinverseX)

Xpadded = helper.paddingOfOnes(X)
leftinverseX = helper.leftInverse(Xpadded)
print("leftinverseX =\n ", leftinverseX)
rightinverseX = helper.rightInverse(Xpadded)
print("rightinverseX =\n", rightinverseX)

print("inv(Xpadded) =\n", inv(Xpadded))

