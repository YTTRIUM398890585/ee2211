import numpy as np
import math

# Please replace "StudentMatriculationNumber" with your actual matric number here and in the filename
def A3_A0257926N(learning_rate, num_iters):
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

    # Function Defs
    def f1(a):
        return a**4
    
    def f1_grad(a):
        return 4*(a**3)
    
    def f2(b):
        return (math.sin(b))**2
    
    def f2_grad(b):
        return 2*math.sin(b)*math.cos(b)
    
    def f3(c,d):
        return (c**5) + d**2 * math.sin(d)
    
    def f3_grad_c(c):
        return 5*(c**4)
    
    def f3_grad_d(d):
        return d * (2*math.sin(d) + d*math.cos(d))

    # Initialization & Parameters
    a = 2.5
    b = 0.6
    c = 2
    d = 3

    a_out = np.zeros(num_iters)
    f1_out = np.zeros(num_iters)
    b_out = np.zeros(num_iters)
    f2_out = np.zeros(num_iters)
    c_out = np.zeros(num_iters)
    d_out = np.zeros(num_iters)
    f3_out = np.zeros(num_iters)

    # Gradient Descent
    for i in range(num_iters):
        a_out[i] = a
        f1_out[i] = f1(a)
        
        b_out[i] = b
        f2_out[i] = f2(b)
        
        c_out[i] = c
        
        d_out[i] = d
        f3_out[i] = f3(c,d)

        a = a - learning_rate*f1_grad(a)
        b = b - learning_rate*f2_grad(b)
        c = c - learning_rate*f3_grad_c(c)
        d = d - learning_rate*f3_grad_d(d)

    # return in this order

    return a_out, f1_out, b_out, f2_out, c_out, d_out, f3_out 
