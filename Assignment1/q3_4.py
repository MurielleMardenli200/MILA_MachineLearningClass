import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from q1_1 import rmse
from q3_1 import compute_gradient_ridge, compute_gradient_simple
from q3_2 import Singleton_Plot_Class, gradient_descent_regression, rmse_with_bias

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


num_epochs = 1000
ridge_hyperparameter = 0.26
learning_rates = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.2]



def plot_learning_rates():
    
    epoch_indexes = list(range(num_epochs))
    singleton_plot_class = Singleton_Plot_Class()

    rmse_for_learning_rates_simple = {}
    rmse_for_learning_rates_ridge = {}

    for learning_rate in learning_rates:
        w_simple, b_simple = gradient_descent_regression(X_train, y_train, reg_type='simple', learning_rate=learning_rate, num_epochs=num_epochs)
        w_ridge, b_ridge = gradient_descent_regression(X_train, y_train, reg_type='ridge', learning_rate=learning_rate, num_epochs=num_epochs, hyperparameter=ridge_hyperparameter)

        rmse_simple = rmse_with_bias(X_test, y_test, w_simple, b_simple)
        rmse_ridge = rmse_with_bias(X_test, y_test, w_ridge, b_ridge, hyperparameter=ridge_hyperparameter)

        singleton_plot_class.plot_lse_per_epoch(epoch_indexes, learning_rate)

        rmse_for_learning_rates_simple[str(learning_rate)] = rmse_simple
        rmse_for_learning_rates_ridge[str(learning_rate)] = rmse_ridge

    plot_rmse_per_learning_rate('simple', rmse_for_learning_rates=rmse_for_learning_rates_simple)
    plot_rmse_per_learning_rate('ridge', rmse_for_learning_rates=rmse_for_learning_rates_ridge)
    plot_rmse_regressions_per_learning_rate((rmse_for_learning_rates_simple, rmse_for_learning_rates_ridge))


def plot_rmse_per_learning_rate(reg_type, rmse_for_learning_rates: dict):

        learning_rates = list(rmse_for_learning_rates.keys())
        rmse_values = list(rmse_for_learning_rates.values())
        plt.plot(learning_rates, rmse_values)

        plt.xlabel('Learning Rate')
        plt.ylabel('RMSE')
        plt.title(f'RMSE According To Learning Rate For {reg_type} Regression')
        plt.grid(True)
        plt.close()
        plt.show()

def plot_rmse_regressions_per_learning_rate(rmse_regressions: tuple):
        simple_learning_rates = list(rmse_regressions[0].keys())
        simple_rmse_values = list(rmse_regressions[0].values())
        ridge_learning_rates = list(rmse_regressions[1].keys())
        ridge_rmse_values = list(rmse_regressions[1].values())

        plt.plot(simple_learning_rates, simple_rmse_values, label='Simple regression')
        plt.plot(ridge_learning_rates, ridge_rmse_values, label = 'Ridge Regression')

        plt.xlabel('Learning Rate')
        plt.ylabel('RMSE')
        plt.title(f'RMSE According To Learning Rate For Both Regressions')
        plt.grid(True)
        plt.legend()
        plt.close()
        plt.show()

plot_learning_rates()