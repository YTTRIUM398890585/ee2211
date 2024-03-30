import sys
sys.path.append('../../')   # nopep8
import ee2211pythonhelper as helper  # nopep8

import numpy as np

import matplotlib.pyplot as plt

from numpy.linalg import inv

import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import OneHotEncoder

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# Question 2 ########################################################################################
# print("Question 2 ########################################################################################")
# X = np.array([[-10], [-8], [-3], [-1], [2], [8]])
# Y = np.array([[5], [5], [4], [3], [2], [2]])

# order = 3
# poly = PolynomialFeatures(order)
# print(poly)
# P = poly.fit_transform(X)
# print("matrix P =\n", P)

# w_poly = inv(P.T @ P) @ P.T @ Y
# print("Estimated w_poly =\n", w_poly)

# Xpadded = helper.paddingOfOnes(X)
# print("Xpadded =\n", Xpadded)
# w_lin = inv(Xpadded.T @ Xpadded) @ Xpadded.T @ Y
# print("Estimated w_lin =\n", w_lin)

# xtest = np.array([[9]])
# xtest_poly = poly.fit_transform(xtest)
# print("xtest_poly =\n", xtest_poly)
# ytest_poly = xtest_poly @ w_poly
# print("ytest_poly =\n", ytest_poly)

# xtest_lin = helper.paddingOfOnes(xtest)
# print("xtest_lin =\n", xtest_lin)
# ytest_lin = xtest_lin @ w_lin
# print("ytest_lin =\n", ytest_lin)

# print("Question 2 tesing helper code ########################################################################################")
# X = np.array([[-10], [-8], [-3], [-1], [2], [8]])
# Y = np.array([[5], [5], [4], [3], [2], [2]])

# order = 3
# w_poly = helper.polynomialRegression(X, Y, order, 0, True)

# w_lin = helper.linearRegressionWithBias(X, Y, True)

# xtest = np.array([[9]])
# ytest_poly = helper.testPolyReg(xtest, w_poly, order, True)

# xtest_lin = helper.paddingOfOnes(xtest)
# ytest_lin = helper.testData(xtest_lin, w_lin, True)

# # Question 3 ########################################################################################
# print("Question 3 ########################################################################################")
# X = np.array([[1, 0, 1], [1, -1, 1]])
# Y = np.array([[0], [1]])

# order = 3
# poly = PolynomialFeatures(order)
# print(poly)
# P = poly.fit_transform(X)
# print("matrix P =\n", P)

# # order 3 with 3 features means 6C3 monomials
# # P is 2x20
# # m < d use dual with no ridge which is just the right inverse
# w_poly = helper.rightInverse(P) @ Y
# print("w_poly =\n", w_poly)

# # or just use linear regression in helper, without bias since we have bias in P already 
# w_poly = helper.linearRegressionWithoutBias(P, Y, printResult=True, printFeature=None)

# # or use dual form with ridge = 0
# w_poly = helper.ridgeRegressionDual(P, Y, 0, True)
1
# # if want to use primal form, need to do with ridge
# w_poly = helper.ridgeRegressionPrimal(P, Y, 0.0001, True)

# w_poly = helper.polynomialRegression(X, Y, order, 0, True)

# # Question 4 ########################################################################################
# print("Question 4 ########################################################################################")
# X = np.array([[-1.0], [0.0], [0.5], [0.3], [0.8]])
# Y = np.array([[1], [1], [2], [1], [2]])

# # encode class1 to 1 and class2 to -1
# Yclass = np.where(Y == 1, 1, -1) 

# w = helper.linearRegressionWithBias(X, Yclass, True)

# Xtest = np.array([[-0.1], [0.4]])
# Ypred = helper.testData(helper.paddingOfOnes(Xtest), w, True)

# YpredClass = np.where(Ypred > 0, 1, 2)
# print("YpredClass =\n", YpredClass)

# Question 5 ########################################################################################
# print("Question 5 ########################################################################################")
# X = np.array([[-1.0], [0.0], [0.5], [0.3], [0.8]])
# Y = np.array([[1], [1], [2], [3], [2]])

# # one hot encoding
# oneHotEncoder = OneHotEncoder(sparse=False)
# print("oneHotEncoder", oneHotEncoder)
# Yencoded = oneHotEncoder.fit_transform(Y)
# print("Yencoded =\n", Yencoded)

# # linear model
# w_lin = helper.linearRegressionWithBias(X, Yencoded, True)
# Xtest = np.array([[-0.1], [0.4]])
# Ypred = helper.testData(helper.paddingOfOnes(Xtest), w_lin, True)

# YpredClass = oneHotEncoder.inverse_transform(Ypred)
# print("YpredClass =\n", YpredClass)

# # polynomial model with order 5
# w_poly = helper.polynomialRegression(X, Yencoded, 5, 0, True)
# Xtest = np.array([[-0.1], [0.4]])
# Ypred = helper.testPolyReg(Xtest, w_poly, 5, True)

# YpredClass = oneHotEncoder.inverse_transform(Ypred)
# print("YpredClass =\n", YpredClass)

# Question 6 ########################################################################################
# print("Question 6 ########################################################################################")
iris_dataset = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state = 0, test_size = 0.26)

# one hot encoding
oneHotEncoder = OneHotEncoder(sparse=False)
YtrainEncoded = oneHotEncoder.fit_transform(y_train.reshape(-1, 1))

w_lin = helper.linearRegressionWithoutBias(X_train, YtrainEncoded, True)

# YpredEncoded = helper.testData(helper.paddingOfOnes(X_test), w_lin, True)
YpredEncoded = helper.testData(X_test, w_lin, True)

Ypred = oneHotEncoder.inverse_transform(YpredEncoded)
print("Ypred =\n", Ypred)

correct = (y_test.reshape(-1, 1) == Ypred)
num_correct = np.sum(correct)
print("correct = ", num_correct, " out of ", len(y_test))

# w_poly = helper.polynomialRegression(X_train, YtrainEncoded, 2, 0, True)

# YpredEncoded = helper.testPolyReg(X_test, w_poly, 2, True)
# Ypred = oneHotEncoder.inverse_transform(YpredEncoded)
# print("Ypred =\n", Ypred)

# correct = (y_test.reshape(-1, 1) == Ypred)
# num_correct = np.sum(correct)
# print("correct = ", num_correct, " out of ", len(y_test))

# linear answer
ans = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 0, 1]]
ans = oneHotEncoder.inverse_transform(ans)
print("ans =\n", ans)

# polynomial answer, all matches
# ans = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0]]
# ans = oneHotEncoder.inverse_transform(ans)
# print("ans =\n", ans)

correct = (ans == Ypred)
num_correct = np.sum(correct)
print("correct = ", num_correct, " out of ", len(y_test))


