import numpy as np

learning_rate = 0.02
num_iters = 10000

# your code goes here
# gradient descent:
# x = x - learning_rate * df(x)/dx

# f1(a) = a^4
# f1'(a) = 4a^3
# final a should be 0, f1(a) should be 0
a0 = 2.5
a_out = []
a_out.append(a0)

# f2(b) = sin^2(b)
# f2'(b) = 2sin(b)cos(b)
# final b should be 0, f2(b) should be 0
b0 = 0.6
b_out = []
b_out.append(b0)

# f3(c, d) = c^5 + d^2 sin(d)
# f3'(c, d) = [5c^4, d^2 cos(d) + 2d sin(d)] 
# take del of f3
# c = c - learning_rate * c component of del(f3(c, d))
# d = d - learning_rate * d component of del(f3(c, d))
# final c should be 0, final d should be 5.087, f3(c, d) should be -24.083
c0 = 2
d0 = 3
c_out = []
d_out = []
c_out.append(c0)
d_out.append(d0)

# gradient descent
for iteration in range(num_iters):
    a_out.append(a_out[iteration] - learning_rate * 4 * pow(a_out[iteration], 3))
    b_out.append(b_out[iteration] - learning_rate * 2 * np.sin(b_out[iteration]) * np.cos(b_out[iteration]))
    c_out.append(c_out[iteration] - learning_rate * 5 * pow(c_out[iteration], 4))
    d_out.append(d_out[iteration] - learning_rate * (pow(d_out[iteration], 2) * np.cos(d_out[iteration]) + 2 * d_out[iteration] * np.sin(d_out[iteration])))

# take away the initial values
a_out = np.array(a_out[1:])
b_out = np.array(b_out[1:])
c_out = np.array(c_out[1:])
d_out = np.array(d_out[1:])

# sub into the functions to get the scalar output values
f1_out = pow(a_out, 4)
f2_out = np.power(np.sin(b_out), 2)
f3_out = pow(c_out, 5) + pow(d_out, 2) * np.sin(d_out)

# print("a_out =\n", a_out)
# print("f1_out =\n", f1_out)

# print("b_out =\n", b_out)
# print("f2_out =\n", f2_out)

print("c_out =\n", c_out)
print("d_out =\n", d_out)
print("f3_out =\n", f3_out)
