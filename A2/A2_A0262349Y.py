import numpy as np
from numpy.linalg import inv
from numpy import argmax
from numpy import count_nonzero

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

# Please replace "MatricNumber" with your actual matric number here and in the filename
def A2_A0262349Y(N):
    """
    Input type
    :N type: int

    Return type
    :X_train type: numpy.ndarray of size (number_of_training_samples, 4)
    :y_train type: numpy.ndarray of size (number_of_training_samples,)
    :X_test type: numpy.ndarray of size (number_of_test_samples, 4)
    :y_test type: numpy.ndarray of size (number_of_test_samples,)
    :Ytr type: numpy.ndarray of size (number_of_training_samples, 3)
    :Yts type: numpy.ndarray of size (number_of_test_samples, 3)
    :Ptrain_list type: List[numpy.ndarray]
    :Ptest_list type: List[numpy.ndarray]
    :w_list type: List[numpy.ndarray]
    :error_train_array type: numpy.ndarray
    :error_test_array type: numpy.ndarray
    """
    # your code goes here
    
    # Load the iris dataset
    iris_dataset = load_iris()
    
    # Split the dataset into training and test sets, random_state is set to 0 to ensure the same output
    # random_state is the seed used by the random number generator
    X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state = N, test_size = 0.7)
    
    # one-hot encoded training and test target numpy matrix
    # 0 = setosa        = [1, 0, 0]
    # 1 = versicolor    = [0, 1, 0]
    # 2 = virginica     = [0, 0, 1]
    Ytr = np.array([[1, 0, 0] if i == 0 else [0, 1, 0] if i == 1 else [0, 0, 1] for i in y_train])
    Yts = np.array([[1, 0, 0] if i == 0 else [0, 1, 0] if i == 1 else [0, 0, 1] for i in y_test])
    
    # list of training and test polynomial matrices for orders 1 to 8
    Ptrain_list = [PolynomialFeatures(order).fit_transform(X_train) for order in range(1, 9)]
    Ptest_list = [PolynomialFeatures(order).fit_transform(X_test) for order in range(1, 9)]
    
    m_list = [Ptrain.shape[0] for Ptrain in Ptrain_list]
    d_list = [Ptrain.shape[1] for Ptrain in Ptrain_list]

    reg_L2 = 0.0001
    
    # if m > d, use primal form: w = inv( (X.T @ X) + lambda * np.identity(X.shape[1]) ) @ X.T @ y
    # else, use dual form: w = X.T @ inv( (X @ X.T) + lambda * np.identity(X.shape[0]) ) @ y
    # multi output ridge regression
    w_list = [inv( (Ptrain.T @ Ptrain) + reg_L2 * np.identity(Ptrain.shape[1]) ) @ Ptrain.T @ Ytr if m_list[order] > d_list[order] else Ptrain.T @ inv( (Ptrain @ Ptrain.T) + reg_L2 * np.identity(Ptrain.shape[0]) ) @ Ytr for order, Ptrain in enumerate(Ptrain_list)]

    # Ypredict = P @ w
    Ypredict_train_list = [Ptrain_list[order-1] @ w_list[order-1] for order in range(1, 9)]
    Ypredict_test_list = [Ptest_list[order-1] @ w_list[order-1] for order in range(1, 9)]

    # ypredict = argmax(Ypredict), convert one-hot encoded matrix to integer
    ypredict_train_list = np.array([[argmax(Ypredict_train_list[order-1][sample]) for sample in range(len(Ypredict_train_list[order-1]))] for order in range(1, 9)])
    ypredict_test_list = np.array([[argmax(Ypredict_test_list[order-1][sample]) for sample in range(len(Ypredict_test_list[order-1]))] for order in range(1, 9)])

    error_train_array = np.array([count_nonzero(ypredict_train_list[order-1] - y_train) for order in range(1, 9)])
    error_test_array = np.array([count_nonzero(ypredict_test_list[order-1] - y_test) for order in range(1, 9)])

    # return in this order
    return X_train, y_train, X_test, y_test, Ytr, Yts, Ptrain_list, Ptest_list, w_list, error_train_array, error_test_array
