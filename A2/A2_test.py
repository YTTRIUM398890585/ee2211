import numpy as np
from numpy.linalg import inv
from numpy import argmax
from numpy import count_nonzero

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

# Load the iris dataset
iris_dataset = load_iris()

# print(iris_dataset)

N = 1
    
# Split the dataset into training and test sets, random_state is set to 0 to ensure the same output
# random_state is the seed used by the random number generator
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state = N, test_size = 0.7)

# print("X_train\n", X_train)
# print("X_test\n", X_test)
# print("y_train\n", y_train)
# print("y_test\n", y_test)

print("Shape X_train: " + str(X_train.shape[0]) + " x " + str(X_train.shape[1]))
print("Shape X_test : " + str(X_test.shape[0]) + " x " + str(X_test.shape[1]))
print("Shape y_train: " + str(y_train.shape[0]) + " x " + str(1))
print("Shape y_test : " + str(y_test.shape[0]) + " x " + str(1))

# one-hot encoded training and test target numpy matrix
# 0 = setosa        = [1, 0, 0]
# 1 = versicolor    = [0, 1, 0]
# 2 = virginica     = [0, 0, 1]
Ytr = np.array([[1, 0, 0] if i == 0 else [0, 1, 0] if i == 1 else [0, 0, 1] for i in y_train])
# print("Ytr\n", Ytr)
print("Shape Ytr: " + str(Ytr.shape[0]) + " x " + str(Ytr.shape[1]))

Yts = np.array([[1, 0, 0] if i == 0 else [0, 1, 0] if i == 1 else [0, 0, 1] for i in y_test])
# print("Yts\n", Yts)
print("Shape Yts: " + str(Yts.shape[0]) + " x " + str(Yts.shape[1]))

# list of training and test polynomial matrices for orders 1 to 8
Ptrain_list = [PolynomialFeatures(order).fit_transform(X_train) for order in range(1, 9)]
# print("Ptrain_list\n", Ptrain_list)
print("checking for orders: ", len(Ptrain_list))
for order in range(1, 9):
    print("Shape Ptrain_list[" + str(order) + "]: " + str(Ptrain_list[order-1].shape[0]) + " x " + str(Ptrain_list[order-1].shape[1]))

Ptest_list = [PolynomialFeatures(order).fit_transform(X_test) for order in range(1, 9)]
# print("Ptest_list\n", Ptest_list)
print("checking for orders: ", len(Ptest_list))
for order in range(1, 9):
    print("Shape Ptest_list[" + str(order) + "]: " + str(Ptest_list[order-1].shape[0]) + " x " + str(Ptest_list[order-1].shape[1]))

m_list = [Ptrain.shape[0] for Ptrain in Ptrain_list]
print("m_list\n", m_list)
d_list = [Ptrain.shape[1] for Ptrain in Ptrain_list]
print("d_list\n", d_list)

reg_L2 = 0.0001

# if m > d, use primal form: w = inv( (X.T @ X) + lambda * np.identity(X.shape[1]) ) @ X.T @ y
# else, use dual form: w = X.T @ inv( (X @ X.T) + lambda * np.identity(X.shape[0]) ) @ y
# multi output ridge regression
w_list = [inv( (Ptrain.T @ Ptrain) + reg_L2 * np.identity(Ptrain.shape[1]) ) @ Ptrain.T @ Ytr if m_list[order] > d_list[order] else Ptrain.T @ inv( (Ptrain @ Ptrain.T) + reg_L2 * np.identity(Ptrain.shape[0]) ) @ Ytr for order, Ptrain in enumerate(Ptrain_list)]
# print("w_list\n", w_list)
for order in range(1, 9):
    print("Shape w_list[" + str(order) + "]: " + str(w_list[order-1].shape[0]) + " x " + str(w_list[order-1].shape[1]))

# Ypredict = P @ w
Ypredict_train_list = [Ptrain_list[order-1] @ w_list[order-1] for order in range(1, 9)]
# print("Ypredict_train_list\n", Ypredict_train_list)
for order in range(1, 9):
    print("Shape Ypredict_train_list[" + str(order) + "]: " + str(Ypredict_train_list[order-1].shape[0]) + " x " + str(Ypredict_train_list[order-1].shape[1]))

Ypredict_test_list = [Ptest_list[order-1] @ w_list[order-1] for order in range(1, 9)]
# print("Ypredict_test_list\n", Ypredict_test_list)
for order in range(1, 9):
    print("Shape Ypredict_test_list[" + str(order) + "]: " + str(Ypredict_test_list[order-1].shape[0]) + " x " + str(Ypredict_test_list[order-1].shape[1]))

# ypredict = argmax(Ypredict), convert one-hot encoded matrix to integer
ypredict_train_list = np.array([[argmax(Ypredict_train_list[order-1][sample]) for sample in range(len(Ypredict_train_list[order-1]))] for order in range(1, 9)])
# print("ypredict_train_list\n", ypredict_train_list)
for order in range(1, 9):
    print("Shape ypredict_train_list[" + str(order) + "]: " + str(ypredict_train_list[order-1].shape[0]) + " x " + str(1))

ypredict_test_list = np.array([[argmax(Ypredict_test_list[order-1][sample]) for sample in range(len(Ypredict_test_list[order-1]))] for order in range(1, 9)])
# print("ypredict_test_list\n", ypredict_test_list)
for order in range(1, 9):
    print("Shape ypredict_test_list[" + str(order) + "]: " + str(ypredict_test_list[order-1].shape[0]) + " x " + str(1))
    
error_train_array = np.array([count_nonzero(ypredict_train_list[order-1] - y_train) for order in range(1, 9)])
error_test_array = np.array([count_nonzero(ypredict_test_list[order-1] - y_test) for order in range(1, 9)])
print("error_train_array\n", error_train_array)
print("error_test_array\n", error_test_array)

print("error_train_array: " + str(error_train_array.shape[0]) + " x " + str(1))
print("error_test_array: " + str(error_test_array.shape[0]) + " x " + str(1))
