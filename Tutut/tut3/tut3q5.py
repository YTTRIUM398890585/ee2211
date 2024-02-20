from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

# mean 30 Ω and standard deviation of 1.8 Ω
mean = 30
std = 1.8

value1 = 28
value2 = 33

# Calculate the CDF for value1 and value2
cdf_value1 = stats.norm.cdf(value1, mean, std)
cdf_value2 = stats.norm.cdf(value2, mean, std)

# The probability of the distribution falling between value1 and value2
# is the difference between the CDF at these points
probability = cdf_value2 - cdf_value1

print("cdf_value1: ", cdf_value1)
print("cdf_value2: ", cdf_value2)
print("probability: ", probability)

x = np.linspace(30 - 10, 30 + 10, 100)
plt.plot(x, stats.norm.pdf(x, mean, std),'r-', lw=5, alpha=0.6, label='norm pdf')
plt.show()  