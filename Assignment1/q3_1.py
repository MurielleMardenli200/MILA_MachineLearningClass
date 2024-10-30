import numpy as np

def compute_gradient_simple(X, y, w, b):
    """
    Compute the gradients of the loss function with respect to w and b for simple linear regression.

    Args:
        X (np.ndarray): Input features matrix of shape (n, m).
        y (np.ndarray): Target vector of shape (n, ).
        w (np.ndarray): Weights vector of shape (m, ).
        b (float): Bias term.

    Returns:
        grad_w (np.ndarray): Gradient with respect to weights.
        grad_b (float): Gradient with respect to bias.
    """
    # Write your code here ...

    if X.size == 0 or len(y) == 0 or len(w) == 0:
        raise ValueError("Input arrays X, y, and w must not be empty.")

    if X.shape[0] != len(y):
        raise ValueError("Number of rows in X must match the length of y.")
    
    if X.shape[1] != len(w):
        raise ValueError("Number of columns in X must match the length of w.")

    n_examples = X.shape[0]
    m_features = X.shape[1]

    grad_w = np.zeros(m_features)
    grad_b = 0.0

    for i in range(n_examples):

        y_pred = np.dot(X[i], w) + b
        y_diff = y_pred - y[i]
        grad_w += (y_diff * X[i])
        grad_b += y_diff


    return grad_w, grad_b


def compute_gradient_ridge(X, y, w, b, lambda_reg):
    """
    Compute the gradients of the loss function with respect to w and b for ridge regression.

    Args:
        X (np.ndarray): Input features matrix of shape (n, m).
        y (np.ndarray): Target vector of shape (n, ).
        w (np.ndarray): Weights vector of shape (m, ).
        b (float): Bias term.
        lambda_reg (float): Regularization parameter.

    Returns:
        grad_w (np.ndarray): Gradient with respect to weights.
        grad_b (float): Gradient with respect to bias.
    """
    # Write your code here ...

    if X.size == 0 or len(y) == 0 or len(w) == 0:
        raise ValueError("Input arrays X, y, and w must not be empty.")

    if X.shape[0] != len(y):
        raise ValueError("Number of rows in X must match the length of y.")
    
    if X.shape[1] != len(w):
        raise ValueError("Number of columns in X must match the length of w.")

    n_examples = X.shape[0]
    m_features = X.shape[1]

    grad_w = np.zeros(m_features)
    grad_b = 0.0

    for i in range(n_examples):

        y_pred = np.dot(X[i], w) + b
        y_diff = y_pred - y[i]
        grad_w += (X[i] *y_diff)
        grad_b += y_diff

    grad_w += 2 * lambda_reg * w
    grad_b += 2 * lambda_reg * b
    
    return grad_w, grad_b


