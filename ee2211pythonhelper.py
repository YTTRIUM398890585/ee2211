import numpy as np
from numpy.linalg import inv
from numpy.linalg import det
from numpy.linalg import matrix_rank

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

import matplotlib.pyplot as plt


def rightInverse(X):
    # X: np.array of m x d matrix, m sample, d features for m < d, wide matrix
    # return: np.array of d x m matrix, d features, m sample, right inverse of X

    # to have right inverse, X must be full row rank, rank = m, or XXT is invertible
    # RI = X.T @ inv(X @ X.T)

    if det(X @ X.T) == 0:
        raise ValueError("[rightInverse]: X @ X.T is singular, no right inverse exists")

    return X.T @ inv(X @ X.T)


def leftInverse(X):
    # X: np.array of m x d matrix, m sample, d features for m > d, tall matrix
    # return: np.array of d x m matrix, d features, m sample, left inverse of X

    # to have left inverse, X must be full colum rank, rank = d, or XTX is invertible
    # LI = inv(X.T @ X) @ X.T

    if det(X.T @ X) == 0:
        raise ValueError("[leftInverse]: X.T @ X is singular, no left inverse exists")

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
        raise ValueError("[linearRegressionWithoutBias]: X and Y should have the same number of samples")
    
    if m > d:
        if printResult : print("[linearRegressionWithoutBias]: m > d, method used = left inverse")
        
        W = leftInverse(X) @ Y 
        
    elif m == d:
        if printResult : print("[linearRegressionWithoutBias]: m = d, method used = inverse")
        
        if det(X) == 0:
            raise ValueError("[linearRegressionWithoutBias]: X is singular, no inverse exists")
        
        W = inv(X) @ Y 
    else:
        if printResult : print("[linearRegressionWithoutBias]: m < d, method used = right inverse")
        
        W = rightInverse(X) @ Y 
        
    if printResult:
        print("[linearRegressionWithoutBias]: rank(X) = ", matrix_rank(X.T @ X))
        print("[linearRegressionWithoutBias]: W =\n", W)
        MSE = mean_squared_error(X @ W, Y)
        print("[linearRegressionWithoutBias]: MSE = ", MSE)

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
        raise ValueError("[linearRegressionWithBias]: X and Y should have the same number of samples")
    
    if m > dpadded:
        if det(X.T @ X) == 0:
            raise ValueError("[linearRegressionWithBias]: X.T @ X is singular, no left inverse exists")
        
        W = inv(X.T @ X) @ X.T @ Y 
        
        if printResult : print("[linearRegressionWithBias]: m > dpadded, method used = left inverse")
    elif m == dpadded:
        if det(X) == 0:
            raise ValueError("[linearRegressionWithBias]: X is singular, no inverse exists")
        
        W = inv(X) @ Y 
        
        if printResult : print("[linearRegressionWithBias]: m = dpadded, method used = inverse")
    else:
        if det(X @ X.T) == 0:
            raise ValueError("[linearRegressionWithBias]: X @ X.T is singular, no right inverse exists")
        
        W = X.T @ inv(X @ X.T) @ Y 
        
        if printResult : print("[linearRegressionWithBias]: m < dpadded, method used = right inverse")

    if printResult:
        print("[linearRegressionWithBias]: rank(X) = ", matrix_rank(X.T @ X))
        print("[linearRegressionWithBias]: W =\n", W)
        MSE = mean_squared_error(X @ W, Y)
        print("[linearRegressionWithBias]: MSE = ", MSE)

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


def ridgeRegressionPrimal(X, Y, lam, printResult=False, printFeature=None):
    # use for m > d

    # X: np.array of m x d matrix, m sample, d features, no bias
    # y: np.array of m x 1 vector, m sample, 1 output
    # printResult: boolean, print the result of w or not
    # printFeature: None or column of x (feature to plot), plot the graph of X and y, and the regression line or not

    # return: np.array of d x 1 vector, d features, 1 output, w = inv(X.T @ X + lam I) @ X.T @ y
    # I is d x d
    W = inv((X.T @ X) + lam * np.identity(X.shape[1])) @ X.T @ Y

    if printResult:
        print("[ridgeRegressionPrimal]: W =\n", W)
        MSE = mean_squared_error(X @ W, Y)
        print("[ridgeRegressionPrimal]: MSE = ", MSE)

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


def ridgeRegressionDual(X, Y, lam, printResult=False, printFeature=None):
    # use for m < d

    # X: np.array of m x d matrix, m sample, d features, no bias
    # y: np.array of m x 1 vector, m sample, 1 output
    # printResult: boolean, print the result of w or not
    # printFeature: None or column of x (feature to plot), plot the graph of X and y, and the regression line or not

    # return: np.array of d x 1 vector, d features, 1 output, w = X.T @ inv(X @ X.T + lam I)  @ y
    # I is m x m
    W = X.T @ inv((X @ X.T) + lam * np.identity(X.shape[0])) @ Y

    if printResult:
        print("[ridgeRegressionDual]: W =\n", W)
        MSE = mean_squared_error(X @ W, Y)
        print("[ridgeRegressionDual]: MSE = ", MSE)

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


def polynomialRegression(X, Y, order, lam, printResult=False):
    # create a polynomial regression model of order order
    # and shove X into the model to get the polynomial features matrix P
    # input to polynomial fit transform doesn't need to pad 1, it automatically does it
    poly = PolynomialFeatures(order)
    print(poly)
    P = poly.fit_transform(X)
    print("[polynomialRegression]: matrix P =\n", P)

    # if m >= d, use primal ridge regression, if lam == 0, its just left inverse
    if P.shape[0] >= P.shape[1]:
        print("[polynomialRegression]: m >= d, method used = primal ridge regression")
        W = ridgeRegressionPrimal(P, Y, lam, printResult=printResult)
    # else m < d, use dual ridge regression, if lam == 0, its just right inverse
    else:
        print("[polynomialRegression]: m < d, method used = dual ridge regression")
        W = ridgeRegressionDual(P, Y, lam, printResult=printResult)
        
    return W


def testPolyReg(X, W, order, printResult=False):
    # fit test X in to the polynomial model with order order
    # and get the output Y
    poly = PolynomialFeatures(order)
    print(poly)
    P = poly.fit_transform(X)
    print("[testPolyReg]: testPoly =\n", P)
    
    return testData(P, W, printResult=printResult)   
    

def testData(X, W, printResult=False):
    # X: np.array of m x d matrix, m sample, d features, no bias
    # W: np.array of d x h vector, d features, h output
    # return Y: np.array of m x h vector, m sample, h output

    if X.shape[1] != W.shape[0]:
        raise ValueError("[testData]: X and W should have the same number of features")

    Y = X @ W

    if printResult:
        print("[testData]: X @ W = Y =\n", Y)

    return Y

def textToMatrix():
    text = input("Enter the matrix in text format: ")
    return [[float(j) for j in i.split()] for i in text.split(';')]

# while(1):
#     try:
#         print("input matrix =\n", textToMatrix())
#     except:
#         print("Error in input matrix")