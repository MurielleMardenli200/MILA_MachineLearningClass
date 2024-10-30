import numpy as np
import matplotlib.pyplot as plt

from q1_1 import linear_regression_predict, linear_regression_optimize, rmse
from q2_1 import ridge_regression_optimize
from q3_1 import compute_gradient_ridge, compute_gradient_simple


def gradient_descent_regression(X, y, reg_type='simple', hyperparameter=0.0, learning_rate=0.01, num_epochs=100):
    """
    Solves regression tasks using full-batch gradient descent.

    Parameters:
    X (np.ndarray): Input feature matrix of shape (n_samples, n_features).
    y (np.ndarray): Target values of shape (n_samples,).
    reg_type (str): Type of regression ('simple' for simple linear, 'ridge' for ridge regression).
    hyperparameter (float): Regularization parameter, used only for ridge regression.
    learning_rate (float): Learning rate for gradient descent.
    num_epochs (int): Number of epochs for gradient descent.

    Returns:
    w (np.ndarray): Final weights after gradient descent optimization.
    b (float): Final bias after gradient descent optimization.
    """

    # Write your code here ...

    singleton_plot_class = Singleton_Plot_Class()

    np.random.seed(42)
    n_features = X.shape[1]
    w = np.random.normal(0, 1, size=n_features)

    b = 0.0
    training_loss_values = []
    training_rmse_values = []
    epoch_indexes = []

    for i in range(num_epochs):
        if reg_type == 'simple':
            gradient_w, gradient_b = compute_gradient_simple(X, y, w, b)
        else:
            gradient_w, gradient_b = compute_gradient_ridge(X, y, w, b, hyperparameter)

        w = w - learning_rate*gradient_w
        b = b - learning_rate*gradient_b

        training_loss = compute_least_square_error(X, y, w, b, hyperparameter)

        training_loss_values.append(training_loss)
        epoch_indexes.append(i)
    
    create_lse_plots(training_loss_values, epoch_indexes, reg_type, learning_rate)
    
    if reg_type == 'simple':
        singleton_plot_class.simple_train_loss = training_loss_values
        singleton_plot_class.simple_train_rmse = training_rmse_values
    else:
        singleton_plot_class.ridge_train_loss = training_loss_values
        singleton_plot_class.ridge_train_rmse = training_rmse_values

    return w, b


def compute_least_square_error(X, y, w, b, hyperparameter=0.0):
    n_examples = X.shape[0]
    n_features = X.shape[1]
    first_sum = 0
    second_sum = 0

    for i in range(n_examples):
        y_diff = (linear_regression_predict(X[i], w) + b) - y[i]
        first_sum += y_diff ** 2 

    for j in range(n_features):
        second_sum += (w[j] ** 2)

    return first_sum + hyperparameter*second_sum


def rmse_with_bias(X, y, w, b, hyperparameter=0.0):
    data_size = X.shape[0]
    loss_error = compute_least_square_error(X, y, w, b, hyperparameter)
    return np.sqrt((2*loss_error)/data_size)



def predict_with_bias(X, w, b):
        n_examples = X.shape[0]
        y_pred = np.zeros(n_examples)
        
        for i in range(n_examples):
            y_pred[i] = np.dot(X[i], w) + b

        return y_pred


def create_lse_plots(training_loss, epoch_indexes, reg_type, learning_rate):
        plt.figure(figsize=(10, 6))
        plt.plot(epoch_indexes, training_loss, linestyle='-', label='LSE')

        plt.xlabel('Epoch iteration', fontsize=12)
        plt.ylabel('Least Squared Error', fontsize=12)
        plt.title(f'LSE according to epoch iteration for {reg_type} regression\nwith learning rate: {learning_rate}', fontsize=14)

        plt.xticks(ticks=range(0, len(epoch_indexes), max(1, len(epoch_indexes)//10)), rotation=45)  # Set x-ticks at intervals
        plt.grid(True)
        plt.close()
        plt.show()

class Singleton_Plot_Class(object):

    simple_train_loss = []
    ridge_train_loss = []

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Singleton_Plot_Class, cls).__new__(cls)
        return cls.instance

    def clear_train_loss(self):
        self.simple_train_loss = []
        self.ridge_train_loss = []

    def plot_lse_per_epoch(self, epoch_indexes, learning_rate):
            plt.figure(figsize=(10, 6))
            
            plt.plot(epoch_indexes, self.simple_train_loss, label=f'Simple Reg Training Loss')
            plt.plot(epoch_indexes, self.ridge_train_loss, label=f'Ridge Reg Training Loss')

            plt.xlabel('Epoch iteration')
            plt.ylabel('Least Squared Error')
            plt.title(f'LSE according to epoch iteration for both regression\nwith learning rate: {learning_rate}')
            plt.xticks(ticks=range(0, len(epoch_indexes), max(1, len(epoch_indexes)//10)), rotation=45)
            plt.grid(True)
            plt.legend()
            plt.close()
            plt.show()

