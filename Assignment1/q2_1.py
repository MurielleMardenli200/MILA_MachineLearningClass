import numpy as np


def ridge_regression_optimize(y: np.ndarray, X: np.ndarray, hyperparameter: float) -> np.ndarray:
    """Optimizes MSE fit of y = Xw with L2 regularization.

    Args:
        y (np.ndarray): Salary, Numpy array of shape [observations, 1].
        X (np.ndarray): Features (e.g., experience, test_score), Numpy array of shape [observations, features].
        hyperparameter (float): Lambda used in L2 regularization.

    Returns:
        np.ndarray: Optimal parameters (w), Numpy array of shape [features, 1].
    """
    # Write your code here
    
    try:
        identity_matrix_size = X.shape[1]
        inverse_sum = np.linalg.inv(np.matmul(X.transpose(), X) + hyperparameter * np.identity(identity_matrix_size))
        w = np.matmul(inverse_sum, np.matmul(X.transpose(), y))

    except np.linalg.LinAlgError as e:
        return print('Could not inverse matrix in ridge regression optimization due to ' + str(e) + ' problem ')

    return w


