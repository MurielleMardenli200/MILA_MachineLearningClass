import numpy as np
import scipy
from scipy.stats import multivariate_normal
import typing

from knn import preprocess_data
from utils import x_train, y_train, x_test, y_test

def gnb_fit_classifier(X: np.ndarray, Y: np.ndarray, smoothing: float=1e-3) -> typing.Tuple:
    """Fits the GNB classifier on the training data

    Args:
        X (np.ndarray): numpy array of shape [num_samples x num_features] containing the training data
        Y (np.ndarray): numpy array of shape [num_samples, ] containing the training labels
        smoothing (float, optional): constant to avoid division by zero. Defaults to 1e-3. Do not change

    Returns:
        prior_probs (typing.List[float]): list of length `num_classes` containing the prior probabilities of the training labels
        means (typing.List[np.ndarray]): list of length `num_classes` containing the means of the batch of samples belonging to a particular label
                                            shape of each element in the list - (num_features, )
        vars (typing.List[np.ndarray]): list of length `num_classes` containing the variances of the batch of samples belonging to a particular label
                                            shape of each element in the list - (num_features, ). Add 'smoothing' to each variance here to prevent nan errors.
    """

    # to set the prior probability of each label by counting the number of times the label appears in
    # training data and normalizing it by the total number of training samples.
    prior_probs = []

    means, vars = [], []

    labels = np.unique(Y)

    ### Implement here

    for c in labels:
        present_labels = (Y == c)
        X_class = X[present_labels]
        
        prior_prob = X_class.shape[0] / X.shape[0] 
        prior_probs.append(prior_prob)

        mean_class = np.mean(X_class, axis=0)
        means.append(mean_class)
        var_class = np.var(X_class, axis=0) + smoothing
        vars.append(var_class)

    return prior_probs, means, vars


def gnb_predict(X: np.ndarray, prior_probs: typing.List[np.ndarray],
                    means: typing.List[np.ndarray], vars: typing.List[np.ndarray], num_classes: int) -> np.ndarray:
    """Computes the predictions of all test samples from the GNB classifier

    Args:
        X (np.ndarray): numpy array of shape [num_samples x features] containing vectorized test images
        prior_probs (typing.List[float]): list of length `num_classes` containing the prior probabilities of the training labels
        means (typing.List[np.ndarray]): list of length `num_classes` containing the means of the batch of samples belonging to a particular label
        vars (typing.List[np.ndarray]): list of length `num_classes` containing the variances of the batch of samples belonging to a particular label
        num_classes (int): int defining the number of classes

    Returns:
        np.ndarray: numpy array of shape (num_samples) containing predictions for each test sample
    """

    num_samples, feature_dim = X.shape

    all_preds = np.zeros((num_samples,  num_classes))

    # HINT: Check out SciPy's multivariate normal documentation and
    # think about which function to use to prevent underflow issues

    ### Implement here

    if (len(prior_probs) != len(means) != len(vars)):
        raise ValueError('Lengths of prior_probs, means and vars are not the same')

    nClasses = len(prior_probs)

    for i in range(nClasses):
        log_likelihood = multivariate_normal.logpdf(X, mean=means[i], cov=vars[i])
        log_prior = np.log(prior_probs[i]) 
        log_posterior = log_likelihood + log_prior
        all_preds[:, i] = log_posterior

    preds = np.argmax(all_preds, axis=1)

    return preds

def gnb_classifier(train_set, train_labels, test_set, test_labels, smoothing=1e-3):
    """Runs the GNB classifier using the above utility functions

    Args:
        train_set (np.ndarray): numpy array of shape [num_samples x num_features] containing the training data
        train_labels (np.ndarray): numpy array of shape [num_samples, ] containing the training labels
        test_set (np.ndarray): numpy array of shape [num_samples x features] containing vectorized test images
        test_labels (np.ndarray): numpy array of shape [num_samples, ] containing the test labels, used to calculate accuracy
        smoothing (float, optional): constant to avoid division by zero. Defaults to 1e-3. Do not change

    Returns:
        float: prediction accuracy of the gnb classifier on the test set in percentage (i.e. multiplied with 100.0)
    """

    num_classes = len(np.unique(y_train))
    ### Implement here. Remember, it is a better coding practice to calculate the number of classes in your data programatically instead of hard coding here.
    
    prior_probs, means, vars = gnb_fit_classifier(train_set, train_labels, smoothing)
    predictions = gnb_predict(test_set, prior_probs, means, vars, num_classes)

    accuracy = (np.equal(predictions, test_labels).sum())/len(test_set) * 100.0


    return accuracy

if __name__ == "__main__":
    train_images, max_val = preprocess_data(x_train)
    test_images, _ = preprocess_data(x_test, max_val)

    del x_train, x_test

    print(f"Training inputs' shape after vectorization: {train_images.shape}")
    print(f"Testing inputs' shape after vectorization: {test_images.shape}")

    n_train_samples = 60000
    n_val_samples = 10000
    n_test_samples = 10000

    # define the training set and labels
    train_set = train_images[:n_train_samples]
    train_labels = y_train[:n_train_samples]
    print(f"Training set shape: {train_set.shape}")

    # define the validation set and labels
    val_set = train_images[-n_val_samples:]
    val_labels = y_train[-n_val_samples:]
    print(f"Validaton set shape: {val_set.shape}")

    # define the test set and labels
    test_set = test_images[:n_test_samples]
    test_labels = y_test[:n_test_samples]
    print(f"Test set shape: {test_set.shape}")

    # test the model!
    val_acc = gnb_classifier(train_set, train_labels, val_set, val_labels)
    print(f"Validation accuracy: {val_acc} %")
    
    test_acc = gnb_classifier(train_set, train_labels, test_set, test_labels)
    print(f"Test accuracy: {test_acc} %")