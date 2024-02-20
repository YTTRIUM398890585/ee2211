import numpy as np

# Please replace "MatricNumber" with your actual matric number here and in the filename


def A1_A0262349Y(X, y):
    """
    Input type
    :X type: numpy.ndarray
    :y type: numpy.ndarray 

    Return type
    :InvXTX type: numpy.ndarray
    :w type: numpy.ndarray

    """

    # your code goes here
    
    # solve Xw = y
    # since input is always X = 5 x 2 and y = 5 x 1, the system is always over-determined
    # only non-singular matrix will be pass in, so X is always full rank and has left inverse (XT X)^-1 XT, 5 x 5, (XT X)^-1 2 x 2
    # least square solution is given by w' = (XT X)^-1 XT y, 2 x 1
    XTX = X.T @ X
    InvXTX = np.linalg.inv(XTX)
    w = InvXTX @ X.T @ y

    # return in this order
    return InvXTX, w
