import sys
sys.path.append('../../')   # nopep8
import ee2211pythonhelper as helper  # nopep8

import numpy as np

import matplotlib.pyplot as plt

from numpy.linalg import inv

import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

''' Example of finding MSE '''
# y1 = np.array([2.1, 1.5, 5.8, 6.1])
# y2 = np.array([9.1, 9.5, 9.8, 12.7, 13.8, 15.9])
# y = np.array([2.1, 1.5, 5.8, 6.1, 9.1, 9.5, 9.8, 12.7, 13.8, 15.9])

# print("overall depth 1", (mean_squared_error(y1, np.mean(y1)*np.ones(len(y1))) * len(y1) + mean_squared_error(y2, np.mean(y2)*np.ones(len(y2))) * len(y2))/len(y))

# helper.customMSE(y)
# y1MSE = helper.customMSE(y1)
# y2MSE = helper.customMSE(y2)
# print("overall depth 1", (y1MSE * len(y1) + y2MSE * len(y2))/len(y))


''' Example for linear regression with biasing and polynomial regression '''
# X = np.array([[-10], [-8], [-3], [-1], [2], [8]])
# Y = np.array([[5], [5], [4], [3], [2], [2]])

# order = 3
# w_poly = helper.polynomialRegression(X, Y, order, 0, forceMethod=None, printResult=True)

# w_lin = helper.linearRegressionWithBias(X, Y, printResult=True)

# xtest = np.array([[9]])
# ytest_poly = helper.testPolyReg(xtest, w_poly, order, True)

# xtest_lin = helper.paddingOfOnes(xtest)
# ytest_lin = helper.testData(xtest_lin, w_lin, True)

# # Plot the original data
# plt.scatter(X, Y, color='black', label='Original data')

# # Add labels and legend
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.legend()

# # Show the plot
# plt.show()

''' Example for pearson correlation coefficient '''
# f1 = np.array([0.3510, 2.1812, 0.2415, -0.1096, 0.1544])
# f2 = np.array([1.1796, 2.1068, 1.7753, 1.2747, 2.0851])
# f3 = np.array([-0.9852, 1.3766, -1.3244, -0.6316, -0.8320])
# y = np.array([0.2758, 1.4392, -0.4611, 0.6154, 1.0006])

# print("Correlation between Feature_1 and Target_y: ", helper.correlation(f1, y))
# print("Correlation between Feature_2 and Target_y: ", helper.correlation(f2, y))
# print("Correlation between Feature_3 and Target_y: ", helper.correlation(f3, y))

'''Q9'''
f1 = np.array([-0.709, 1.7255, 0.9539, -0.7581, -1.035, -1.049])
f2 = np.array([2.8719, 1.5014, 1.8365, -0.5467, 1.8274, 0.3501])
f3 = np.array([-1.8349, 0.4055, 1.0118, 0.5171, 0.7279, 1.2654])
f4 = np.array([2.6354, 2.7448, 1.4616, 0.7258, -1.6893, -1.7512])
y = np.array([0.8206, 1.0639, 0.6895, -0.0252, 0.995, 0.6608])

print("Correlation between Feature_1 and Target_y: ", helper.correlation(f1, y))
print("Correlation between Feature_2 and Target_y: ", helper.correlation(f2, y))
print("Correlation between Feature_3 and Target_y: ", helper.correlation(f3, y))
print("Correlation between Feature_4 and Target_y: ", helper.correlation(f4, y))

''' Example for polynomial regression with ridge and dual method'''
# # training data
# X_train = np.array([-10, -8, -3, -1, 2, 7]).reshape((-1, 1))
# Y_train = np.array([4.18, 2.42, 0.22, 0.12, 0.25, 3.09]).reshape((-1, 1))

# # test data
# X_test = np.array([-9, -7, -5, -4, -2, 1, 4, 5, 6, 9]).reshape((-1, 1))
# Y_test = np.array([3, 1.81, 0.80, 0.25, -0.19, 0.4, 1.24, 1.68, 2.32, 5.05]).reshape((-1, 1))

# print("shape of X_train: ", X_train.shape)
# print("shape of Y_train: ", Y_train.shape)
# print("shape of X_test: ", X_test.shape)
# print("shape of Y_test: ", Y_test.shape)

# MSE = [("MSE_train_noRidge", "MSE_test_noRidge", "MSE_train_Ridge", "MSE_test_Ridge", "MSE_train_Ridge_forceDual", "MSE_test_Ridge_forceDual")]

# for order in range (1, 7):
#     print("Order: ", order)
    
#     # no ridge
#     w = helper.polynomialRegression(X_train, Y_train, order, 0, printResult=True)
#     print("shape of w: ", w.shape)
    
#     Ytr_pred = helper.testPolyReg(X_train, w, order, True)
#     MSE_train_noRidge = mean_squared_error(Ytr_pred, Y_train)

#     Yts_pred = helper.testPolyReg(X_test, w, order, True)
#     MSE_test_noRidge = mean_squared_error(Yts_pred, Y_test)
    
#     # with ridge
#     w = helper.polynomialRegression(X_train, Y_train, order, 1, printResult=True)
#     print("shape of w: ", w.shape)
    
#     Ytr_pred = helper.testPolyReg(X_train, w, order, True)
#     MSE_train_Ridge = mean_squared_error(Ytr_pred, Y_train)

#     Yts_pred = helper.testPolyReg(X_test, w, order, True)
#     MSE_test_Ridge = mean_squared_error(Yts_pred, Y_test)
    
#     # with ridge and force the method to dual
#     w = helper.polynomialRegression(X_train, Y_train, order, 1, forceMethod="dual", printResult=True)
#     print("shape of w: ", w.shape)
    
#     Ytr_pred = helper.testPolyReg(X_train, w, order, True)
#     MSE_train_Ridge_forceDual = mean_squared_error(Ytr_pred, Y_train)

#     Yts_pred = helper.testPolyReg(X_test, w, order, True)
#     MSE_test_Ridge_forceDual = mean_squared_error(Yts_pred, Y_test)
    
#     MSE.append((MSE_train_noRidge, MSE_test_noRidge, MSE_train_Ridge, MSE_test_Ridge, MSE_train_Ridge_forceDual, MSE_test_Ridge_forceDual))

# for MSE in MSE:
#     print(MSE)

''' Example for gradient descent '''
# learning_rate = 0.01
# num_iters = 10

# print(helper.gradientDescentApprox(lambda a:a**4, 2.5, learning_rate, num_iters)[0])
# print(helper.gradientDescentApprox(lambda a:a**4, 2.5, learning_rate, num_iters)[1])

''' Q22 '''

# learning_rate = 0.1
# num_iters = 5

# print(helper.gradientDescent(lambda a:np.sin(np.exp(a))**3, lambda a:3*np.cos(np.exp(a))*np.exp(a)*np.sin(np.exp(a))**2, 3, learning_rate, num_iters)[0])
# print(helper.gradientDescent(lambda a:np.sin(np.exp(a))**3, lambda a:3*np.cos(np.exp(a))*np.exp(a)*np.sin(np.exp(a))**2, 3, learning_rate, num_iters)[1])
# print(helper.gradientDescent(lambda a:np.sin(np.exp(a))**3, lambda a:3*np.cos(np.exp(a))*np.exp(a)*np.sin(np.exp(a))**2, 3, learning_rate, num_iters)[2])


'''Q29'''
# learning_rate = 0.03
# num_iters = 5

# print(helper.gradientDescent(lambda xyz:xyz[1]*xyz[0]**2 + xyz[1]**3 + xyz[0]*xyz[1]*xyz[2], lambda xyz:(2*xyz[0]*xyz[1] + xyz[1]*xyz[2], xyz[0]**2 + 3*xyz[1]**2 + xyz[0]*xyz[2], xyz[0]*xyz[1]), (2, 6, -3), learning_rate, num_iters)[0])
# print(helper.gradientDescent(lambda xyz:xyz[1]*xyz[0]**2 + xyz[1]**3 + xyz[0]*xyz[1]*xyz[2], lambda xyz:(2*xyz[0]*xyz[1] + xyz[1]*xyz[2], xyz[0]**2 + 3*xyz[1]**2 + xyz[0]*xyz[2], xyz[0]*xyz[1]), (2, 6, -3), learning_rate, num_iters)[1])


# print(helper.gradientDescentApprox(lambda b:np.sin(b)**2, 0.6, learning_rate, num_iters)[0])
# print(helper.gradientDescentApprox(lambda b:np.sin(b)**2, 0.6, learning_rate, num_iters)[1])

# print(helper.gradientDescent(lambda cd:cd[0]**5 + (cd[1]**2) * np.sin(cd[1]), lambda cd:(5*cd[0]**4, (2*cd[1]) * np.sin(cd[1]) + (cd[1]**2) * np.cos(cd[1])), (2, 3), learning_rate, num_iters)[0])
# print(helper.gradientDescent(lambda cd:cd[0]**5 + (cd[1]**2) * np.sin(cd[1]), lambda cd:(5*cd[0]**4, (2*cd[1]) * np.sin(cd[1]) + (cd[1]**2) * np.cos(cd[1])), (2, 3), learning_rate, num_iters)[1])

# # by experiment, the best smallStep is 1e-1
# # its best if can input fprime manually, then check with approximation, scalar function seems to be fine
# print(helper.gradientDescentApprox(lambda cd:cd[0]**5 + (cd[1]**2) * np.sin(cd[1]), (2, 3), learning_rate, num_iters, smallStep=1e-1)[0])
# print(helper.gradientDescentApprox(lambda cd:cd[0]**5 + (cd[1]**2) * np.sin(cd[1]), (2, 3), learning_rate, num_iters, smallStep=1e-1)[1])

''' Example for tree classifier '''
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split

# iris_dataset = load_iris()

# X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], 
#                                                         iris_dataset['target'], 
#                                                         test_size=0.20, 
#                                                         random_state=0)

# print("X_train shape: ", X_train.shape)
# print("X_test shape: ", X_test.shape)
# print("y_train shape: ", y_train.shape)
# print("y_test shape: ", y_test.shape)

# print(y_test)

# helper.treeClassifier(X_train, X_test, y_train, y_test, 'entropy', 4)
# helper.treeClassifier(X_train, X_test, y_train, y_test, 'gini', 4)

''' Example for tree regressor '''
# S = np.array([3.2, 3.8, 0.5, 0.6, 1.0, 2.0, 3.0])
# P = np.array([0.75, 0.80, 0.19, 0.23, 0.28, 0.42, 0.53])

# helper.treeRegressor(S[:5], S[5:], P[:5], P[5:], 'squared_error', 1)
# helper.treeRegressor(S, S[:1], P, P[:1], 'squared_error', 1)
