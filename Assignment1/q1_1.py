import numpy as np

# Part A:
def data_matrix_bias(X: np.ndarray) -> np.ndarray:
    """Returns the design matrix with an all one column appended

    Args:
        X (np.ndarray): Numpy array of shape [observations, num_features]

    Returns:
        np.ndarray: Numpy array of shape [observations, num_features + 1]
    """

    if len(X) == 0:
        raise ValueError("Input matrix X is empty.")
    
    X_bias_column = np.ones((X.shape[0], 1), dtype=X.dtype)
    X_bias = np.hstack((X_bias_column, X))

    return X_bias

def data_matrix_B(X: np.ndarray, B_bias) -> np.ndarray:
    """Returns the design matrix with a random column appended

    Args:
        X (np.ndarray): Numpy array of shape [observations, num_features]

    Returns:
        np.ndarray: Numpy array of shape [observations, num_features + 1]
    """


    B = np.ones((X.shape[0], 1), dtype=X.dtype) * B_bias
    X_bias = np.hstack((B, X))

    return X_bias

# Part B:
def linear_regression_predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Computes $y = Xw$

    Args:
        X (np.ndarray): Numpy array of shape [observations, features]
        w (np.ndarray): Numpy array of shape [features, 1]

    Returns:
        np.ndarray: Numpy array of shape [observations, 1]
    """
    return np.matmul(X, w)


# Part C:
def linear_regression_optimize(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Optimizes MSE fit of $y = Xw$

    Args:
        y (np.ndarray): Numpy array of shape [observations, 1]
        X (np.ndarray): Numpy array of shape [observations, features]

    Returns:
        Numpy array of shape [features, 1]
    """

    if len(X)==0 or len(y)==0:
        raise ValueError("Input matrix X or vector y is empty.")
    
    if X.shape[0] != len(y):
        raise ValueError(f"Shape mismatch: X has {X.shape[0]} observations but y has {len(y)}.")

    try:
        X_transposed = np.transpose(X)
        inverse_matrix = np.linalg.inv(np.matmul(X_transposed, X))

        second_multiplication = np.matmul(X_transposed, y)
        optimal_w = np.matmul(inverse_matrix, second_multiplication)

    except np.linalg.LinAlgError as e:
        raise ValueError('Could not optimize linear regression paramater due to ' + str(e) + ' problem ')
        
    return optimal_w


# Part D
def rmse(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Evaluate the RMSE between actual and predicted values.

    Parameters:
    y (list or np.array): The actual values.
    y_hat (list or np.array): The predicted values.

    Returns:
    float: The RMSE value.
    """

    if len(y) == 0 or len(y_hat) == 0:
        raise ValueError("One or both input arrays are empty.")
    
    if len(y) != len(y_hat):
        raise ValueError(f"Shape mismatch for y and y_hat")

    try: 
        summation = np.mean( np.subtract(y, y_hat) ** 2)
        rmse_err = np.sqrt(summation)
        return rmse_err

    except Exception as e:
        raise ValueError('Could not compute RMSE due to error: ' + str(e))

