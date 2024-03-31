import numpy as np

import sys
sys.path.append('../../')   # nopep8
import ee2211pythonhelper as helper  # nopep8
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import PolynomialFeatures


Feature_1 = [0.3510, 2.1812, 0.2415, -0.1096, 0.1544]
Feature_2 = [1.1796, 2.1068, 1.7753, 1.2747, 2.0851]
Feature_3 = [-0.9852, 1.3766, -1.3244, -0.6316, -0.8320]
Target_y = [0.2758, 1.4392, -0.4611, 0.6154, 1.0006]

# mean_Feature_1 = sum(Feature_1)/len(Feature_1)
# mean_Feature_2 = sum(Feature_2)/len(Feature_2)
# mean_Feature_3 = sum(Feature_3)/len(Feature_3)
# mean_Target_y = sum(Target_y)/len(Target_y)

# sd_Feature_1 = (sum([(i - mean_Feature_1)**2 for i in Feature_1])/len(Feature_1))**0.5
# sd_Feature_2 = (sum([(i - mean_Feature_2)**2 for i in Feature_2])/len(Feature_2))**0.5
# sd_Feature_3 = (sum([(i - mean_Feature_3)**2 for i in Feature_3])/len(Feature_3))**0.5
# sd_Target_y = (sum([(i - mean_Target_y)**2 for i in Target_y])/len(Target_y))**0.5

# cov_Feature_1_Target_y = sum([(Feature_1[i] - mean_Feature_1)*(Target_y[i] - mean_Target_y) for i in range(len(Feature_1))])/len(Feature_1)
# cov_Feature_2_Target_y = sum([(Feature_2[i] - mean_Feature_2)*(Target_y[i] - mean_Target_y) for i in range(len(Feature_2))])/len(Feature_2)
# cov_Feature_3_Target_y = sum([(Feature_3[i] - mean_Feature_3)*(Target_y[i] - mean_Target_y) for i in range(len(Feature_3))])/len(Feature_3)

# corr_Feature_1_Target_y = cov_Feature_1_Target_y/(sd_Feature_1*sd_Target_y)
# corr_Feature_2_Target_y = cov_Feature_2_Target_y/(sd_Feature_2*sd_Target_y)
# corr_Feature_3_Target_y = cov_Feature_3_Target_y/(sd_Feature_3*sd_Target_y)

# print("Correlation between Feature_1 and Target_y: ", corr_Feature_1_Target_y)
# print("Correlation between Feature_2 and Target_y: ", corr_Feature_2_Target_y)
# print("Correlation between Feature_3 and Target_y: ", corr_Feature_3_Target_y)

# def correlation(Feature, Target_y):
#     mean_Feature = sum(Feature)/len(Feature)
#     mean_Target_y = sum(Target_y)/len(Target_y)
    
#     sd_Feature = (sum([(i - mean_Feature)**2 for i in Feature])/len(Feature))**0.5
#     sd_Target_y = (sum([(i - mean_Target_y)**2 for i in Target_y])/len(Target_y))**0.5
    
#     cov_Feature_Target_y = sum([(Feature[i] - mean_Feature)*(Target_y[i] - mean_Target_y) for i in range(len(Feature))])/len(Feature)
    
#     return cov_Feature_Target_y/(sd_Feature*sd_Target_y)

# print("Correlation between Feature_1 and Target_y: ", helper.correlation(Feature_1, Target_y))
# print("Correlation between Feature_2 and Target_y: ", helper.correlation(Feature_2, Target_y))
# print("Correlation between Feature_3 and Target_y: ", helper.correlation(Feature_3, Target_y))


# f1 = np.array([0.3510, 2.1812, 0.2415, -0.1096, 0.1544])
# f2 = np.array([1.1796, 2.1068, 1.7753, 1.2747, 2.0851])
# f3 = np.array([-0.9852, 1.3766, -1.3244, -0.6316, -0.8320])
# y = np.array([0.2758, 1.4392, -0.4611, 0.6154, 1.0006])

# print("Correlation between Feature_1 and Target_y: ", helper.correlation(f1, y))
# print("Correlation between Feature_2 and Target_y: ", helper.correlation(f2, y))
# print("Correlation between Feature_3 and Target_y: ", helper.correlation(f3, y))

''' Q2 '''
# training data
X_train = np.array([-10, -8, -3, -1, 2, 7]).reshape((-1, 1))
Y_train = np.array([4.18, 2.42, 0.22, 0.12, 0.25, 3.09]).reshape((-1, 1))

# test data
X_test = np.array([-9, -7, -5, -4, -2, 1, 4, 5, 6, 9]).reshape((-1, 1))
Y_test = np.array([3, 1.81, 0.80, 0.25, -0.19, 0.4, 1.24, 1.68, 2.32, 5.05]).reshape((-1, 1))

print("shape of X_train: ", X_train.shape)
print("shape of Y_train: ", Y_train.shape)
print("shape of X_test: ", X_test.shape)
print("shape of Y_test: ", Y_test.shape)

MSE = [("MSE_train_noRidge", "MSE_test_noRidge", "MSE_train_Ridge", "MSE_test_Ridge", "MSE_train_Ridge_forceDual", "MSE_test_Ridge_forceDual")]

for order in range (1, 30):
    print("Order: ", order)
    
    w = helper.polynomialRegression(X_train, Y_train, order, 0, printResult=True)
    print("shape of w: ", w.shape)
    
    Ytr_pred = helper.testPolyReg(X_train, w, order, True)
    MSE_train_noRidge = mean_squared_error(Ytr_pred, Y_train)

    Yts_pred = helper.testPolyReg(X_test, w, order, True)
    MSE_test_noRidge = mean_squared_error(Yts_pred, Y_test)
    
    w = helper.polynomialRegression(X_train, Y_train, order, 1, printResult=True)
    print("shape of w: ", w.shape)
    
    Ytr_pred = helper.testPolyReg(X_train, w, order, True)
    MSE_train_Ridge = mean_squared_error(Ytr_pred, Y_train)

    Yts_pred = helper.testPolyReg(X_test, w, order, True)
    MSE_test_Ridge = mean_squared_error(Yts_pred, Y_test)
    
    w = helper.polynomialRegression(X_train, Y_train, order, 1, forceMethod="dual", printResult=True)
    print("shape of w: ", w.shape)
    
    Ytr_pred = helper.testPolyReg(X_train, w, order, True)
    MSE_train_Ridge_forceDual = mean_squared_error(Ytr_pred, Y_train)

    Yts_pred = helper.testPolyReg(X_test, w, order, True)
    MSE_test_Ridge_forceDual = mean_squared_error(Yts_pred, Y_test)
    
    MSE.append((MSE_train_noRidge, MSE_test_noRidge, MSE_train_Ridge, MSE_test_Ridge, MSE_train_Ridge_forceDual, MSE_test_Ridge_forceDual))

for MSE in MSE:
    print(MSE)
    
