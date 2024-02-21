import numpy as np
from numpy.linalg import inv
from numpy.linalg import det
from numpy.linalg import matrix_rank

from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

def rightInverse(X):
    # X: np.array of m x d matrix, m sample, d features for m < d, wide matrix
    # return: np.array of d x m matrix, d features, m sample, right inverse of X
    
    # to have right inverse, X must be full row rank, rank = m, or XXT is invertible
    # RI = X.T @ inv(X @ X.T)
    
    if det(X @ X.T) == 0:
        raise ValueError("X @ X.T is singular, no right inverse exists")

    return X.T @ inv(X @ X.T)

def leftInverse(X):
    # X: np.array of m x d matrix, m sample, d features for m > d, tall matrix
    # return: np.array of d x m matrix, d features, m sample, left inverse of X
    
    # to have left inverse, X must be full colum rank, rank = d, or XTX is invertible
    # LI = inv(X.T @ X) @ X.T
    
    if det(X.T @ X) == 0:
        raise ValueError("X.T @ X is singular, no left inverse exists")

    return inv(X.T @ X) @ X.T

def paddingOfOnes(X):
    # X: np.array of m x d matrix, m sample, d features, no bias
    # return: np.array of m x (d+1) matrix, m sample, d+1 features, with bias
    
    return np.hstack((np.ones((X.shape[0], 1)), X))


def linearRegressionWithoutBias(X, Y, printResult=False, printFeature=None):
    # X: np.array of m x d matrix, m sample, d features, no bias
    # Y: np.array of m x h vector, m sample, h output
    # printResult: boolean, print the result of rank(X.T @ X) and w or not
    # printGraph: None or column of x (feature to plot), plot the graph of X and Y, and the regression line or not

    # return: np.array of d x h vector, d features, h output, w = inv(X.T @ X) @ X.T @ Y

    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y should have the same number of samples")

    if det(X.T @ X) == 0:
        raise ValueError("X.T @ X is singular, no inverse exists")

    W = inv(X.T @ X) @ X.T @ Y

    if printResult:
        print("rank(X) = ", matrix_rank(X.T @ X))
        print("W =\n", W)
        MSE = mean_squared_error(X @ W,Y)
        print("MSE = ", MSE)

    if printFeature != None:
        # X[:, printFeature] = all rows in column printFeature
        # plot the graph of printFeature column (zero index) of X and Y, original
        plt.plot(X[:, printFeature], Y, 'o', label = 'original')

        # plot the graph of printFeature column (zero index) of X and XW, regression
        plt.plot(X[:, printFeature], X @ W, '-', label = 'regression')
        
        plt.xlabel('X')
        plt.ylabel('Y')
        
        plt.show()

    return W

def linearRegressionWithBias(X, Y, printResult=False, printFeature=None):
    # X: np.array of m x d matrix, m sample, d features, no bias
    # Y: np.array of m x h vector, m sample, h output
    # printResult: boolean, print the result of rank(X.T @ X) and w or not
    # printGraph: None or column of x (feature to plot with bias), plot the graph of X and Y, and the regression line or not

    # return: np.array of d x h vector, d features, h output, w = inv(X.T @ X) @ X.T @ Y
    
    # add bias to X
    X = paddingOfOnes(X)

    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y should have the same number of samples")

    if det(X.T @ X) == 0:
        raise ValueError("X.T @ X is singular, no inverse exists")

    W = inv(X.T @ X) @ X.T @ Y

    if printResult:
        print("rank(X) = ", matrix_rank(X.T @ X))
        print("W =\n", W)
        MSE = mean_squared_error(X @ W,Y)
        print("MSE = ", MSE)

    if printFeature != None:
        # X[:, printFeature] = all rows in column printFeature
        # plot the graph of printFeature column (zero index) of X and Y, original
        plt.plot(X[:, printFeature], Y, 'o', label = 'original')

        # plot the graph of printFeature column (zero index) of X and XW, regression
        plt.plot(X[:, printFeature], X @ W, '-', label = 'regression')
        
        plt.xlabel('X')
        plt.ylabel('Y')
        
        plt.show()

    return W

def testData(X, W, printResult=False):
    # X: np.array of m x d matrix, m sample, d features, no bias
    # W: np.array of d x h vector, d features, h output
    # return Y: np.array of m x h vector, m sample, h output
    
    if X.shape[1] != W.shape[0]:
        raise ValueError("X and W should have the same number of features")
    
    Y = X @ W
    
    if printResult:
        print("X @ W = Y =\n", Y)

    return Y
    