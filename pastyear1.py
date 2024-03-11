import sys
sys.path.append('../../')   # nopep8
import ee2211pythonhelper as helper  # nopep8

import numpy as np

import matplotlib.pyplot as plt

from numpy.linalg import inv

import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

X = np.array([[1, 2], [0, 6], [1, 0], [0, 5], [1, 7]])
Y = np.array([[1], [2], [3], [4], [5]])

w = helper.linearRegressionWithBias(X, Y, printResult=True, printFeature=None)

xtest = np.array([[1, 3]])
xtest = helper.paddingOfOnes(xtest)
ytest = helper.testData(xtest, w, True)

X = np.array([[3], [4], [5], [6], [7]])
Y = np.array([[5], [4], [3], [2], [1]])

w = helper.linearRegressionWithBias(X, Y, printResult=True, printFeature=None)

xtest = np.array([[8]])
xtest = helper.paddingOfOnes(xtest)
ytest = helper.testData(xtest, w, True)

