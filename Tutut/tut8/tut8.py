import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append('../../')   # nopep8
import ee2211pythonhelper as helper  # nopep8

''' Question 1 '''
# print(helper.gradientDescentApprox(lambda x:x**4, 2, 0.1, 1)[0])
# print(helper.gradientDescent(lambda x:x**4, lambda x:4*x**3, 2, 0.1, 1)[0])

''' Question 2 '''
# read in data and reshape to column vectors
df = pd.read_csv("C:/University Stuff/Y2S2/EE2211/Tutut/tut8/GovernmentExpenditureonEducation.csv")
expenditure = df['total_expenditure_on_education'].to_numpy().reshape(-1, 1)
years = df['year'].to_numpy().reshape(-1, 1)

# normalise the data
expenditure = expenditure/np.max(expenditure)
years = years/np.max(years)

years = helper.paddingOfOnes(years)

sample = expenditure.shape[0]

# print(expenditure)
# print(years)

# xw = (x, w), matrix x = years (sample x 2), w = expenditure (2 x 1)
# model returns a matrix of size (sample x 1)
# gradient descent take cost as row vector (1 x 2)
model = lambda xw: np.exp(-xw[0] @ xw[1][:, np.newaxis])

# return scalar 
cost = lambda w: (model((years, w)) - expenditure).T @ (model((years, w)) - expenditure)

# return gradient of cost wrt w of size (2 x 1)
# gradient descent take cost as row vector (1 x 2)
# costPrime = lambda w: (- 2 * sum((model((years[i], w)) - expenditure[i])*(model((years[i], w)))*(years[i]) for i in range(sample)))
costPrime = lambda w: - 2 * (model((years, w)) - expenditure).T @ (model((years, w))) @ (years)

learning_rate = 0.03
num_iters = 2000000

initial_w = np.array((0, 0))

print(costPrime(initial_w))

# (a) Plot the cost function C(w) as a function of the number of iterations
w, c = helper.gradientDescent(cost, costPrime, initial_w, learning_rate, num_iters)
# because of 
c = c.flatten()
iterations = np.arange(0, num_iters + 1)

plt.plot(iterations, c)
plt.xlabel('Number of Iterations')
plt.ylabel('Cost Function C(w)')
plt.title('Cost Function C(w) vs. Number of Iterations')
plt.show()

# # (b) Use the fitted parameters to plot the predicted educational expenditure from year 1981 to year 2023
# years_pred = np.arange(1981, 2024).reshape(-1, 1)
# years_pred = helper.paddingOfOnes(years_pred)
# expenditure_pred = model((years_pred, w_optimal))

# plt.plot(years_pred, expenditure_pred)
# plt.xlabel('Year')
# plt.ylabel('Predicted Educational Expenditure')
# plt.title('Predicted Educational Expenditure from 1981 to 2023')
# plt.show()

# # (c) Repeat (a) using a learning rate of 0.1
# learning_rate_0_1 = 0.1
# w_optimal_0_1 = helper.gradientDescent(cost, costPrime, initial_w, learning_rate_0_1, num_iters)[0]
# cost_values_0_1 = [cost(w) for w in helper.gradientDescentPath(cost, costPrime, initial_w, learning_rate_0_1, num_iters)]

# plt.plot(iterations, cost_values_0_1)
# plt.xlabel('Number of Iterations')
# plt.ylabel('Cost Function C(w)')
# plt.title('Cost Function C(w) vs. Number of Iterations (Learning Rate = 0.1)')
# plt.show()

# # Repeat (a) using a learning rate of 0.001
# learning_rate_0_001 = 0.001
# w_optimal_0_001 = helper.gradientDescent(cost, costPrime, initial_w, learning_rate_0_001, num_iters)[0]
# cost_values_0_001 = [cost(w) for w in helper.gradientDescentPath(cost, costPrime, initial_w, learning_rate_0_001, num_iters)]

# plt.plot(iterations, cost_values_0_001)
# plt.xlabel('Number of Iterations')
# plt.ylabel('Cost Function C(w)')
# plt.title('Cost Function C(w) vs. Number of Iterations (Learning Rate = 0.001)')
# plt.show()







