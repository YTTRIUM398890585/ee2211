import numpy as np
from numpy.linalg import inv
from numpy.linalg import det
from numpy.linalg import matrix_rank

# question 1
print("question 1")
X = np.array([[1, 1], [3, 4]])
y = np.array([0, 1])
print("det(X) = ", det(X))
print("rank(X) = ", matrix_rank(X))
w = inv(X) @ y
print("w = ", w)

# question 2
print("question 2")
X = np.array([[1, 2], [3, 6]])
y = np.array([0, 1])
print("det(X) = ", det(X))
print("rank(X) = ", matrix_rank(X))
# w = inv(X) @ y
# print("w: ", w)

# question 3
print("question 3")
X = np.array([[1, 2], [2, 4], [1, -1]])
y = np.array([0, 0.1, 1])
print("rank(X): ", matrix_rank(X))
w = inv(X.T @ X) @ X.T @ y  # find using left inverse ==> least square solution
print("w = ", w)

# question 4
print("question 4")
X = np.array([[1, 0, 1, 0], [1, -1, 1, -1], [1, 1, 0, 0]])
y = np.array([1, 0, 1])
print("rank(X): ", matrix_rank(X))
w = X.T @ inv(X @ X.T) @ y  # find using right inverse ==> least norm solution
print("w = ", w)

# question 6
print("question 6")
X = np.array([[1, 2], [2, 4], [1, -1]])
y = np.array([0, 1])
XT = X.T
print("rank(XT): ", matrix_rank(XT))

# wT @ X = y ==> XT @ w = y, XT is wide matrix, uses right inverse
w = XT.T @ inv(XT @ XT.T) @ y  # find using right inverse ==> least norm solution
print("w = ", w)