import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_pdf import PdfPages
from q1_1 import rmse
from q3_1 import compute_gradient_ridge, compute_gradient_simple
from q3_2 import gradient_descent_regression, Singleton_Plot_Class, predict_with_bias, rmse_with_bias
 
# Load the dataset
X_train = pd.read_csv('Data/X_train.csv').values
y_train = pd.read_csv('Data/y_train.csv').values
X_test = pd.read_csv('Data/X_test.csv').values
y_test = pd.read_csv('Data/y_test.csv').values

X_train = (X_train-np.min(X_train))/(np.max(X_train)-np.min(X_train))
y_train = y_train.reshape(-1)
y_train = (y_train-np.min(y_train))/(np.max(y_train)-np.min(y_train))

X_test = (X_test-np.min(X_test))/(np.max(X_test)-np.min(X_test))
y_test = y_test.reshape(-1)
y_test = (y_test-np.min(y_test))/(np.max(y_test)-np.min(y_test))


learning_rate = 0.01
num_epochs = 10000
ridge_hyperparameter = 0.26


# Provide your code here ...

singleton_plot_class = Singleton_Plot_Class()
singleton_plot_class.clear_train_loss()
w_lin_reg, b_lin_reg = gradient_descent_regression(X_train, y_train, reg_type='simple', learning_rate=learning_rate, num_epochs=num_epochs)
w_ridge_reg, b_ridge_reg = gradient_descent_regression(X_train, y_train, reg_type='ridge', learning_rate=learning_rate, num_epochs=num_epochs, hyperparameter=ridge_hyperparameter)

epoch_indexes = list(range(num_epochs))
singleton_plot_class.plot_lse_per_epoch(epoch_indexes, learning_rate)

y_pred_simple = predict_with_bias(X_test, w_lin_reg, b_lin_reg)
y_pred_ridge = predict_with_bias(X_test, w_ridge_reg, b_ridge_reg)

rmse_simple = rmse_with_bias(X_test, y_test, w_lin_reg, b_lin_reg)
rmse_ridge = rmse_with_bias(X_test, y_test, w_ridge_reg, b_ridge_reg, hyperparameter=ridge_hyperparameter)


plt.figure(figsize=(10, 6)) 
labels = ['Simple', 'Ridge'] 
plt.bar(labels, [rmse_simple, rmse_ridge], color=['lightblue', 'lightgreen'])

for i, v in enumerate([rmse_simple, rmse_ridge]):
    plt.text(i + 0.05, v, f'{v:.2f}', va='center') 

plt.xlabel('Type of Regression')
plt.ylabel('RMSE')
plt.close()
plt.show()

