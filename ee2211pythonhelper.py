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
    # printFeature: None or column of x (feature to plot), plot the graph of X and Y, and the regression line or not

    # return: np.array of d x h vector, d features, h output, 
    # if X is over determined, m > d, use left inverse: w = inv(X.T @ X) @ X.T @ Y 
    # if X is even determined, m = d, use inverse: w = inv(X) @ Y 
    # if X is under determined, m < d, use right inverse: w = X.T @ inv(X @ X.T) @ Y 
    
    m = X.shape[0]
    d = X.shape[1]

    if m != Y.shape[0]:
        raise ValueError("X and Y should have the same number of samples")
    
    if m > d:
        if det(X.T @ X) == 0:
            raise ValueError("X.T @ X is singular, no left inverse exists")
        
        W = inv(X.T @ X) @ X.T @ Y 
        
        if printResult : print("m > d, methodUsed = left inverse")
    elif m == d:
        if det(X) == 0:
            raise ValueError("X is singular, no inverse exists")
        
        W = inv(X) @ Y 
        
        if printResult : print("m = d, methodUsed = inverse")
    else:
        if det(X @ X.T) == 0:
            raise ValueError("X @ X.T is singular, no right inverse exists")
        
        W = X.T @ inv(X @ X.T) @ Y 
        
        if printResult : print("m < d, methodUsed = right inverse")
        
    if printResult:
        print("rank(X) = ", matrix_rank(X.T @ X))
        print("W =\n", W)
        MSE = mean_squared_error(X @ W, Y)
        print("MSE = ", MSE)

    if printFeature != None:
        # X[:, printFeature] = all rows in column printFeature
        # plot the graph of printFeature column (zero index) of X and Y, original
        plt.plot(X[:, printFeature], Y, 'o', label='original')

        # plot the graph of printFeature column (zero index) of X and XW, regression
        plt.plot(X[:, printFeature], X @ W, '-', label='regression')

        plt.xlabel('X')
        plt.ylabel('Y')

        plt.show()

    return W


def linearRegressionWithBias(X, Y, printResult=False, printFeature=None):
    # X: np.array of m x d matrix, m sample, d features, no bias
    # Y: np.array of m x h vector, m sample, h output
    # printResult: boolean, print the result of rank(X.T @ X) and w or not
    # printFeature: None or column of x (feature to plot with bias), plot the graph of X and Y, and the regression line or not

    # return: np.array of d x h vector, d features, h output, 
    # if X is over determined, m > d, use left inverse: w = inv(X.T @ X) @ X.T @ Y 
    # if X is even determined, m = d, use inverse: w = inv(X) @ Y 
    # if X is under determined, m < d, use right inverse: w = X.T @ inv(X @ X.T) @ Y 

    # add bias to X
    X = paddingOfOnes(X)
    
    # check for size after padding
    m = X.shape[0]
    dpadded = X.shape[1]

    if m != Y.shape[0]:
        raise ValueError("X and Y should have the same number of samples")
    
    if m > dpadded:
        if det(X.T @ X) == 0:
            raise ValueError("X.T @ X is singular, no left inverse exists")
        
        W = inv(X.T @ X) @ X.T @ Y 
        
        if printResult : print("m > dpadded, methodUsed = left inverse")
    elif m == dpadded:
        if det(X) == 0:
            raise ValueError("X is singular, no inverse exists")
        
        W = inv(X) @ Y 
        
        if printResult : print("m = dpadded, methodUsed = inverse")
    else:
        if det(X @ X.T) == 0:
            raise ValueError("X @ X.T is singular, no right inverse exists")
        
        W = X.T @ inv(X @ X.T) @ Y 
        
        if printResult : print("m < dpadded, methodUsed = right inverse")

    if printResult:
        print("rank(X) = ", matrix_rank(X.T @ X))
        print("W =\n", W)
        MSE = mean_squared_error(X @ W, Y)
        print("MSE = ", MSE)

    if printFeature != None:
        # X[:, printFeature] = all rows in column printFeature
        # plot the graph of printFeature column (zero index) of X and Y, original
        plt.plot(X[:, printFeature], Y, 'o', label='original')

        # plot the graph of printFeature column (zero index) of X and XW, regression
        plt.plot(X[:, printFeature], X @ W, '-', label='regression')

        plt.xlabel('X')
        plt.ylabel('Y')

        plt.show()

    return W


def ridgeRegressionPrimal(X, y, lam, printResult=False, printFeature=None):
    # use for m > d

    # X: np.array of m x d matrix, m sample, d features, no bias
    # y: np.array of m x 1 vector, m sample, 1 output
    # printResult: boolean, print the result of w or not
    # printFeature: None or column of x (feature to plot), plot the graph of X and y, and the regression line or not

    # return: np.array of d x 1 vector, d features, 1 output, w = inv(X.T @ X + lam I) @ X.T @ y
    # I is d x d
    w = inv((X.T @ X) + lam * np.identity(X.shape[1])) @ X.T @ y

    if printResult:
        print("W =\n", W)

    if printFeature != None:
        # X[:, printFeature] = all rows in column printFeature
        # plot the graph of printFeature column (zero index) of X and Y, original
        plt.plot(X[:, printFeature], y, 'o', label='original')

        # plot the graph of printFeature column (zero index) of X and XW, regression
        plt.plot(X[:, printFeature], X @ w, '-', label='regression')

        plt.xlabel('X')
        plt.ylabel('y')

        plt.show()

    return w


def ridgeRegressionDual(X, y, lam, printResult=False, printFeature=None):
    # use for m < d

    # X: np.array of m x d matrix, m sample, d features, no bias
    # y: np.array of m x 1 vector, m sample, 1 output
    # printResult: boolean, print the result of w or not
    # printFeature: None or column of x (feature to plot), plot the graph of X and y, and the regression line or not

    # return: np.array of d x 1 vector, d features, 1 output, w = X.T @ inv(X @ X.T + lam I)  @ y
    # I is m x m
    w = X.T @ inv((X @ X.T) + lam * np.identity(X.shape[0])) @ y

    if printResult:
        print("W =\n", W)

    if printFeature != None:
        # X[:, printFeature] = all rows in column printFeature
        # plot the graph of printFeature column (zero index) of X and Y, original
        plt.plot(X[:, printFeature], y, 'o', label='original')

        # plot the graph of printFeature column (zero index) of X and XW, regression
        plt.plot(X[:, printFeature], X @ w, '-', label='regression')

        plt.xlabel('X')
        plt.ylabel('y')

        plt.show()

    return w


# def polynomialRegression(X, y, order, printResult=False, printFeature=None):

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
