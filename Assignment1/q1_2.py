import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import os


from q1_1 import data_matrix_bias, linear_regression_predict, linear_regression_optimize, rmse, data_matrix_B
from q2_1 import ridge_regression_optimize

# Loading the dataset
X_train = pd.read_csv('Data/X_train.csv').values
y_train = pd.read_csv('Data/y_train.csv').values
X_test = pd.read_csv('Data/X_test.csv').values
y_test = pd.read_csv('Data/y_test.csv').values


X_train_biased = data_matrix_bias(X_train)
optimal_parameters = linear_regression_optimize(y_train, X_train_biased)
y_train_prediction = linear_regression_predict(X_train_biased, optimal_parameters)

X_test_biased = data_matrix_bias(X_test)
y_test_prediction = linear_regression_predict(X_test_biased, optimal_parameters)

rmse_value = rmse(y_test, y_test_prediction)


# Q2.5: B column bias
B_bias = 10000000
X_train_B = data_matrix_B(X_train, B_bias)
optimal_parameters_B = ridge_regression_optimize(y_train, X_train_B, hyperparameter=0.1)
print('\noptimal parameters with a Bias of 1')
print(optimal_parameters)
print(f'\noptimal parameters with a Bias of {B_bias}')
print(optimal_parameters_B)
    

X_test_experience = X_test[:, 0]
X_test_test_score = X_test[:, 1]

plt.scatter(X_test_experience, y_test, label='Real Salary')
plt.scatter(X_test_experience, y_test_prediction, label='Predicted Salary')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.title('Salary prediction according to experience')
plt.legend()
plt.grid(True)
plt.close()

plt.scatter(X_test_test_score, y_test, label='Real Salary')
plt.scatter(X_test_test_score, y_test_prediction, label='Predicted Salary')
plt.xlabel('Test score')
plt.ylabel('Salary')
plt.title('Salary prediction according to test score')
plt.legend()
plt.grid(True)
plt.close()

plt.show()