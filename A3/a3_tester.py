import A3_A0262349Y as grading

import A3_A0257926N as gradingJSIM

import numpy as np

import sys
sys.path.append('../')   # nopep8
import ee2211pythonhelper as helper  # nopep8

learning_rate = 0.01
num_iters = 10
a_out, f1_out, b_out, f2_out, c_out, d_out, f3_out = grading.A3_A0262349Y(learning_rate, num_iters)

print("a_out = \n", a_out)
print("f1_out = \n", f1_out)
# print("Length of a_out = ", len(a_out))
# print("Length of f1_out = ", len(f1_out))
# print("Type of a_out = ", type(a_out))
# print("Type of f1_out = ", type(f1_out))

print("b_out = \n", b_out)
print("f2_out = \n", f2_out)
# print("Length of b_out = ", len(b_out))
# print("Length of f2_out = ", len(f2_out))
# print("Type of b_out = ", type(b_out))
# print("Type of f2_out = ", type(f2_out))

print("c_out = \n", c_out)
print("d_out = \n", d_out)
print("f3_out = \n", f3_out)
# print("Length of c_out = ", len(c_out))
# print("Length of d_out = ", len(d_out))
# print("Length of f3_out = ", len(f3_out))
# print("Type of c_out = ", type(c_out))
# print("Type of d_out = ", type(d_out))
# print("Type of f3_out = ", type(f3_out))


print("__________________________JSIM_________________________")

a_out, f1_out, b_out, f2_out, c_out, d_out, f3_out = gradingJSIM.A3_A0257926N(learning_rate, num_iters)

print("a_out = \n", a_out)
print("f1_out = \n", f1_out)
# print("Length of a_out = ", len(a_out))
# print("Length of f1_out = ", len(f1_out))
# print("Type of a_out = ", type(a_out))
# print("Type of f1_out = ", type(f1_out))

print("b_out = \n", b_out)
print("f2_out = \n", f2_out)
# print("Length of b_out = ", len(b_out))
# print("Length of f2_out = ", len(f2_out))
# print("Type of b_out = ", type(b_out))
# print("Type of f2_out = ", type(f2_out))

print("c_out = \n", c_out)
print("d_out = \n", d_out)
print("f3_out = \n", f3_out)
# print("Length of c_out = ", len(c_out))
# print("Length of d_out = ", len(d_out))
# print("Length of f3_out = ", len(f3_out))
# print("Type of c_out = ", type(c_out))
# print("Type of d_out = ", type(d_out))
# print("Type of f3_out = ", type(f3_out))

print("__________________________HELPER_________________________")


print(helper.gradientDescentApprox(lambda a:a**4, 2.5, learning_rate, num_iters)[0])
print(helper.gradientDescentApprox(lambda a:a**4, 2.5, learning_rate, num_iters)[1])

print(helper.gradientDescentApprox(lambda b:np.sin(b)**2, 0.6, learning_rate, num_iters)[0])
print(helper.gradientDescentApprox(lambda b:np.sin(b)**2, 0.6, learning_rate, num_iters)[1])

print(helper.gradientDescent(lambda cd:cd[0]**5 + (cd[1]**2) * np.sin(cd[1]), lambda cd:(5*cd[0]**4, (2*cd[1]) * np.sin(cd[1]) + (cd[1]**2) * np.cos(cd[1])), (2, 3), learning_rate, num_iters)[0])
print(helper.gradientDescent(lambda cd:cd[0]**5 + (cd[1]**2) * np.sin(cd[1]), lambda cd:(5*cd[0]**4, (2*cd[1]) * np.sin(cd[1]) + (cd[1]**2) * np.cos(cd[1])), (2, 3), learning_rate, num_iters)[1])

# by experiment, the best smallStep is 1e-1
# its best if can input fprime manually, then check with approximation, scalar function seems to be fine
print(helper.gradientDescentApprox(lambda cd:cd[0]**5 + (cd[1]**2) * np.sin(cd[1]), (2, 3), learning_rate, num_iters, smallStep=1e-1)[0])
print(helper.gradientDescentApprox(lambda cd:cd[0]**5 + (cd[1]**2) * np.sin(cd[1]), (2, 3), learning_rate, num_iters, smallStep=1e-1)[1])
