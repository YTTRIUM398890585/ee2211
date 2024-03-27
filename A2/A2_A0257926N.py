import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures

def A2_A0257926N(N):
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
    # load iris data
    TEST_SIZE = 0.7
    iris_data, iris_target = load_iris(as_frame=True, return_X_y=True)

    # split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(iris_data, iris_target, test_size=TEST_SIZE, random_state=N)

    # convert target to one-hot encoding
    onehot_encoder = OneHotEncoder(sparse_output=False)
    Ytr = onehot_encoder.fit_transform(y_train.values.reshape(-1, 1))
    Yts = onehot_encoder.transform(y_test.values.reshape(-1, 1))

    # Init lists to store results of diff orders
    Ptrain_list = []
    Ptest_list = []
    w_list = []
    error_train_array = []
    error_test_array = []

    for order in range(1,9):
        # Create polynomial features
        poly = PolynomialFeatures(degree=order)
        Ptrain = poly.fit_transform(X_train)
        Ptrain_list.append(Ptrain)
        Ptest = poly.fit_transform(X_test)
        Ptest_list.append(Ptest)

        # Find number of cols and rows
        num_of_rows = Ptrain.shape[0]
        num_of_cols = Ptrain.shape[1]

        # Apply ridge regression with regulatization
        LAMBDA = 0.0001
        if num_of_rows <= num_of_cols:
            # Dual form ridge regression
            r2 = LAMBDA * np.identity(num_of_rows)
            w = Ptrain.T @ np.linalg.inv(Ptrain @ Ptrain.T + r2) @ Ytr
        else:
            # Primal form ridge regression
            r2 = LAMBDA * np.identity(num_of_cols)
            w = np.linalg.inv(Ptrain.T @ Ptrain + r2) @ Ptrain.T @ Ytr
        w_list.append(w)

        # Predict outputs for training and test sets
        y_train_pred = np.argmax(Ptrain @ w, axis=1)
        y_test_pred = np.argmax(Ptest @ w, axis=1)

        # Calculate error counts
        error_train = np.sum(y_train_pred != y_train)
        error_train_array.append(error_train)
        error_test = np.sum(y_test_pred != y_test)
        error_test_array.append(error_test)

    # return in this order
    return X_train, y_train, X_test, y_test, Ytr, Yts, Ptrain_list, Ptest_list, w_list, error_train_array, error_test_array