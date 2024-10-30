
import numpy as np
from typing import List, Tuple
import math

from q1_1 import linear_regression_predict, rmse
from q2_1 import ridge_regression_optimize


def is_last_fold(index: int, k_folds: int):
    return index == k_folds - 1

def cross_validation_linear_regression(k_folds: int, hyperparameters: List[float],
                                       X: np.ndarray, y: np.ndarray) -> Tuple[float, float, List[float]]:
    """
    Perform k-fold cross-validation to find the best hyperparameter for Ridge Regression.

    Args:
        k_folds (int): Number of folds to use.
        hyperparameters (List[float]): List of floats containing the hyperparameter values to search.
        X (np.ndarray): Numpy array of shape [observations, features].
        y (np.ndarray): Numpy array of shape [observations, 1].

    Returns:
        best_hyperparam (float): Value of the best hyperparameter found.
        best_mean_squared_error (float): Best mean squared error corresponding to the best hyperparameter.
        mean_squared_errors (List[float]): List of mean squared errors for each hyperparameter.
    """

    # Write your code here ...
    data_size = X.shape[0]
    fold_size = data_size / k_folds
    mean_squared_errors = np.array([])

    for hyperparameter in hyperparameters: 
        rmse_sum = 0

        for i in range(k_folds):

            rounded_fold_index = math.floor(fold_size*(i))
            upper_bound = math.floor(fold_size*(i+1))
            
            if(is_last_fold(i, k_folds)):
                X_val = X[rounded_fold_index : (data_size-1)]
                y_val = y[rounded_fold_index : (data_size-1)]
                X_train = X[:rounded_fold_index]
                y_train = y[:rounded_fold_index]
            
            else:
                # Manages the case where X[1:1] = []
                if rounded_fold_index == upper_bound: 
                    X_val = X[rounded_fold_index]
                    y_val = y[rounded_fold_index]
                else:
                    X_val = X[rounded_fold_index : upper_bound]
                    y_val = y[rounded_fold_index : upper_bound]
                X_train = np.concatenate((X[:rounded_fold_index], X[upper_bound:]))
                y_train = np.concatenate((y[:rounded_fold_index], y[upper_bound:]))
            
            optimal_parameters = np.array([])
            optimal_parameters = ridge_regression_optimize(y_train, X_train, hyperparameter)

            y_val_prediction = linear_regression_predict(X_val, optimal_parameters)
            rmse_value = rmse(y_val, y_val_prediction)
            rmse_sum += rmse_value
        
        rmse_average = rmse_sum / k_folds
        mean_squared_errors = np.append(mean_squared_errors, rmse_average)

    best_mean_squared_error = np.min(mean_squared_errors)
    best_hyperparam_index = np.where(mean_squared_errors == best_mean_squared_error)[0][0]
    best_hyperparam = hyperparameters[best_hyperparam_index] 

    return best_hyperparam, best_mean_squared_error, mean_squared_errors.tolist()