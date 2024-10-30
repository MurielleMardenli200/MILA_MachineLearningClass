from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from q1_1 import data_matrix_bias
from q2_2 import cross_validation_linear_regression

# Define a range of alpha values for hyperparameter search
hyperparams = np.logspace(-4, 4, 50)
X_train = pd.read_csv('Data/X_train.csv').values
y_train = pd.read_csv('Data/y_train.csv').values
X_test = pd.read_csv('Data/X_test.csv').values
y_test = pd.read_csv('Data/y_test.csv').values
kfolds = 5

# Write your code here ...
X_train = data_matrix_bias(X_train)
best_hyperparam, best_mean_squared_error, mean_squared_errors = cross_validation_linear_regression(kfolds, hyperparams, X_train, y_train)



plt.plot(hyperparams, mean_squared_errors)
plt.xlabel('Hyperparameters')
plt.ylabel('Root Mean square error (RMSE)')
plt.title('Root Mean Square Error According to Hyperparameter')
plt.grid(True)
plt.close()
plt.show()
