import A1_A0262349Y as a
import numpy as np

X = np.array([[1, 1], [4, 2], [5, 6], [3, -6], [0, -10]])
y = np.array([-3, 2, 1, 6, 4])

InvXTX, w = a.A1_A0262349Y(X, y)
print("InvXTX = \n", InvXTX)
print()
print("w = \n", w)
# InvXTX = 
#  [[ 0.02061495 -0.00244584]
#  [-0.00244584  0.0059399 ]]

# w =
#  [ 0.74598183 -0.47833683]

# >> inv(transpose(X)*X)

# ans =

#     0.0206   -0.0024
#    -0.0024    0.0059

# >> inv(transpose(X)*X)*transpose(X)*y

# ans =

#     0.7460
#    -0.4783

X = np.array([[1, 1], [2, 2], [3, 6], [4, -8], [0, 55]])
y = np.array([12345, 7654, 480, 642, 555])

InvXTX, w = a.A1_A0262349Y(X, y)
print("InvXTX = \n", InvXTX)
print()
print("w = \n", w)
# InvXTX =
#  [[3.33621122e-02 9.59293960e-05]
#  [9.59293960e-05 3.19764653e-04]]

# w =
#  [1061.64239653   20.91909954]

# >> inv(transpose(X)*X)

# ans =

#     0.0334    0.0001
#     0.0001    0.0003

# >> inv(transpose(X)*X)*transpose(X)*y

# ans =

#    1.0e+03 *

#     1.0616
#     0.0209

X = np.array([[51, 1], [1, 52], [0, 0], [2, 1], [-100, 100]])
y = np.array([98, 55, 4, 10, -1000])

InvXTX, w = a.A1_A0262349Y(X, y)
print("InvXTX = \n", InvXTX)
print()
print("w = \n", w)
# InvXTX =
#  [[0.00020408 0.00015893]
#  [0.00015893 0.00020247]]

# w =
#  [ 6.02186017 -2.94708749]

# >> inv(transpose(X)*X)

# ans =

#    1.0e-03 *

#     0.2041    0.1589
#     0.1589    0.2025

# >> inv(transpose(X)*X)*transpose(X)*y

# ans =

#     6.0219
#    -2.9471

# crashes if X is singular
# X = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [-55, -55]])
# y = np.array([12345, 7654, 480, 642, 555])

# InvXTX, w = a.A1_A0262349Y(X, y)
# print("InvXTX = \n", InvXTX)
# print()
# print("w = \n", w)