import numpy as np
from numpy.linalg import inv
from numpy.linalg import det
from numpy.linalg import matrix_rank

import matplotlib.pyplot as plt


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
        print("W = ", W)

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

X = np.array([[3], [4], [10], [6], [7]])
Y = np.array([[0, 5], [1.5, 4], [-3, 8], [-4, 10], [1, 6]])
w = linearRegressionWithoutBias(X, Y, printResult=False, printFeature=0)
print(w)

def linearRegressionWithBias(X, Y, printResult=False, printFeature=None):
    # X: np.array of m x d matrix, m sample, d features, no bias
    # Y: np.array of m x h vector, m sample, h output
    # printResult: boolean, print the result of rank(X.T @ X) and w or not
    # printGraph: None or column of x (feature to plot), plot the graph of X and Y, and the regression line or not

    # return: np.array of d x h vector, d features, h output, w = inv(X.T @ X) @ X.T @ Y
    
    # add bias to X
    X = np.hstack((np.ones((X.shape[0], 1)), X))

    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y should have the same number of samples")

    if det(X.T @ X) == 0:
        raise ValueError("X.T @ X is singular, no inverse exists")

    W = inv(X.T @ X) @ X.T @ Y

    if printResult:
        print("rank(X) = ", matrix_rank(X.T @ X))
        print("W = ", W)

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

X = np.array([[3], [4], [10], [6], [7]])
Y = np.array([[0, 5], [1.5, 4], [-3, 8], [-4, 10], [1, 6]])
w = linearRegressionWithBias(X, Y, printResult=False, printFeature=1)
print(w)
