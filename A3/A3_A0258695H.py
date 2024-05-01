import numpy as np
from math import sin
from math import cos

# Please replace "StudentMatriculationNumber" with your actual matric number here and in the filename
def A3_A0258695H(learning_rate, num_iters):
    """
    Input type
    :learning_rate type: float
    :num_iters type: int

    Return type
    :a_out type: numpy array of length num_iters
    :f1_out type: numpy array of length num_iters
    :b_out type: numpy array of length num_iters
    :f2_out type: numpy array of length num_iters
    :c_out type: numpy array of length num_iters
    :d_out type: numpy array of length num_iters
    :f3_out type: numpy array of length num_iters
    """
    # your code goes here
    a = 2.5
    b = 0.6
    c = 2
    d = 3
    a_out = np.array([])
    f1_out = np.array([])
    b_out = np.array([])
    f2_out = np.array([])
    c_out = np.array([])
    d_out = np.array([])
    f3_out = np.array([])
    L = learning_rate
    for i in range (num_iters):
        a = a - L * 4 * a ** 3
        a_out = np.append(a_out, a)
        f1 = a ** 4
        f1_out = np.append(f1_out, f1)
        b = b - L * 2 * sin(b) * cos(b)
        b_out = np.append(b_out, b)
        f2 = sin(b) ** 2
        f2_out = np.append(f2_out, f2)
        c = c - L * 5 * c ** 4
        d = d - L * (2 * d * sin(d) + d ** 2 * cos(d))
        c_out = np.append(c_out, c)
        d_out = np.append(d_out, d)
        f3 = c ** 5 + d ** 2 * sin(d)
        f3_out = np.append(f3_out, f3)
    # return in this order
    return a_out, f1_out, b_out, f2_out, c_out, d_out, f3_out 