import numpy as np
from numpy.linalg import inv
from numpy.linalg import det
from numpy.linalg import matrix_rank

from sklearn import tree
from sklearn.metrics import mean_squared_error, accuracy_score
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


def polynomialRegression(X, Y, order, lam, forceMethod=None, printResult=False):
    # create a polynomial regression model of order order
    # and shove X into the model to get the polynomial features matrix P
    # input to polynomial fit transform doesn't need to pad 1, it automatically does it
    poly = PolynomialFeatures(order)
    P = poly.fit_transform(X)
    
    if printResult:
        print("[polynomialRegression]: matrix P =\n", P)
        
    if forceMethod == None:
        # if m >= d, use primal ridge regression, if lam == 0, its just left inverse
        if P.shape[0] >= P.shape[1]:
            if printResult:
                print("[polynomialRegression]: m >= d, method used = primal ridge regression")
                
            W = ridgeRegressionPrimal(P, Y, lam, printResult=printResult)
        # else m < d, use dual ridge regression, if lam == 0, its just right inverse
        else:
            if printResult:
                print("[polynomialRegression]: m < d, method used = dual ridge regression")
                
            W = ridgeRegressionDual(P, Y, lam, printResult=printResult)
    elif forceMethod == "dual":
        if printResult:
                print("[polynomialRegression]: forced to use dual ridge regression")
                
        W = ridgeRegressionDual(P, Y, lam, printResult=printResult)
    elif forceMethod == "primal":
        if printResult:
                print("[polynomialRegression]: forced to use primal ridge regression")
                
        W = ridgeRegressionPrimal(P, Y, lam, printResult=printResult)
        
    return W


def testPolyReg(X, W, order, printResult=False):
    # fit test X in to the polynomial model with order order
    # and get the output Y
    poly = PolynomialFeatures(order)
    P = poly.fit_transform(X)
    
    if printResult:
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

def correlation(Feature, Target_y):
    mean_Feature = sum(Feature)/len(Feature)
    mean_Target_y = sum(Target_y)/len(Target_y)
    
    sd_Feature = (sum([(i - mean_Feature)**2 for i in Feature])/len(Feature))**0.5
    sd_Target_y = (sum([(i - mean_Target_y)**2 for i in Target_y])/len(Target_y))**0.5
    
    cov_Feature_Target_y = sum([(Feature[i] - mean_Feature)*(Target_y[i] - mean_Target_y) for i in range(len(Feature))])/len(Feature)
    
    return cov_Feature_Target_y/(sd_Feature*sd_Target_y)

def gradientDescentApprox(f, initial, learning_rate, num_iters, smallStep=1e-6):
    # print("smallStep= ", smallStep)
    
    def f_prime(x, smallStep=smallStep):        
        if type(x) == np.float64 or type(x) == np.int32 or len(x) == 1:
            # if scalar function, this performs df/dx
            # print("scalar derivative")
            return (f(x + smallStep) - f(x - smallStep))/(2 * smallStep)
        else:
            # if vector function, this performs del f 
            # print("vector derivative")
            dim = len(x)
            grad = np.zeros(dim)
            for i in range(dim):
                x_plus_h = np.copy(x)
                x_plus_h[i] += smallStep
                x_minus_h = np.copy(x)
                x_minus_h[i] -= smallStep
                grad[i] = (f(x_plus_h) - f(x_minus_h)) / (2 * smallStep)
                
            # print("grad= ", grad)
            return grad
    
    return gradientDescent(f, f_prime, initial, learning_rate, num_iters)

def gradientDescent(f, f_prime, initial, learning_rate, num_iters):
    steps = np.array([initial],)
        
    for iteration in range(num_iters):
        new_step = steps[iteration] - learning_rate * np.array(f_prime(steps[iteration]))
        steps = np.vstack((steps, new_step))
        
    flist = np.array([f(i) for i in steps])

    return steps, flist

def treeRegressor(X_train, X_test, y_train, y_test, criterion, max_depth):
    '''
    Only from 1D to 1D regression
    '''

    # X_train is of shape (n_samples,)
    # X_test is of shape (n_samples,)
    
    # encoded in 0, 1, 2, ...
    # y_train is of shape (n_samples,)
    # y_test is of shape (n_samples,)
    
    sort_index = X_train.argsort()
    X_train = X_train[sort_index]
    y_train = y_train[sort_index]
    
    sort_index = X_test.argsort()
    X_test = X_test[sort_index]
    y_test = y_test[sort_index]

    # scikit decision tree regressor
    scikit_tree = tree.DecisionTreeRegressor(criterion=criterion, max_depth=max_depth)
    scikit_tree.fit(X_train.reshape(-1,1), y_train) # reshape necessary because tree expects 2D array
    
    # predict
    y_trainpred = scikit_tree.predict(X_train.reshape(-1,1))
    y_testpred = scikit_tree.predict(X_test.reshape(-1,1))
    
    # print accuracies
    print("[treeRegressor] Training MSE: ", mean_squared_error(y_train, y_trainpred))
    print("[treeRegressor] Test MSE: ", mean_squared_error(y_test, y_testpred))
    
    # Plot
    plt.figure(0, figsize=[9,4.5])
    plt.rcParams.update({'font.size': 16})
    plt.scatter(X_train, y_train, c='steelblue', s=30)
    plt.plot(X_train, y_trainpred, color='red', lw=2, label='scikit-learn')
    plt.xlabel('X train')
    plt.ylabel('Y train and predict')
    plt.legend(loc='upper right',ncol=3, fontsize=10)
    plt.show()
    
def treeClassifier(X_train, X_test, y_train, y_test, criterion, max_depth):
    # can be any number of features
    # X_train is of shape (n_samples, n_features)
    # X_test is of shape (n_samples, n_features)
    
    # encoded in 0, 1, 2, ...
    # y_train is of shape (n_samples,)
    # y_test is of shape (n_samples,)
    
    # fit tree
    dtree = tree.DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
    dtree = dtree.fit(X_train, y_train) 
    
    # predict
    y_trainpred = dtree.predict(X_train)
    y_testpred = dtree.predict(X_test)
    
    # print accuracies
    print("[treeClassifier] Training accuracy: ", accuracy_score(y_train, y_trainpred))
    print("[treeClassifier] Test accuracy: ", accuracy_score(y_test, y_testpred))    

    # Plot tree
    tree.plot_tree(dtree)
    plt.show()
    
    # tree decision is given with respect to X[feature index]
    # a plot of the tree with [class 0, class 1, class 2, ...] as the amount in each class