import sys
sys.path.append('../../')   # nopep8
import ee2211pythonhelper as helper  # nopep8

import numpy as np

import matplotlib.pyplot as plt

from numpy.linalg import inv

import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures


# Question 2 ########################################################################################
print("Question 2 ########################################################################################")
X = np.array([[-10], [-8], [-3], [-1], [2], [8]])
Y = np.array([[5], [5], [4], [3], [2], [2]])

order = 3
poly = PolynomialFeatures(order)
print(poly)
P = poly.fit_transform(X)
print("matrix P =\n", P)

w_poly = inv(P.T @ P) @ P.T @ Y
print("Estimated w_poly =\n", w_poly)

Xpadded = helper.paddingOfOnes(X)
print("Xpadded =\n", Xpadded)
w_lin = inv(Xpadded.T @ Xpadded) @ Xpadded.T @ Y
print("Estimated w_lin =\n", w_lin)

xtest = np.array([[9]])
xtest_poly = poly.fit_transform(xtest)
print("xtest_poly =\n", xtest_poly)
ytest_poly = xtest_poly @ w_poly
print("ytest_poly =\n", ytest_poly)

xtest_lin = helper.paddingOfOnes(xtest)
print("xtest_lin =\n", xtest_lin)
ytest_lin = xtest_lin @ w_lin
print("ytest_lin =\n", ytest_lin)




