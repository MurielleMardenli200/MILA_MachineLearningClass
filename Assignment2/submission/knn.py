import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.spatial

from utils import x_train, y_train, x_test, y_test

# NOTE: Testing done
def preprocess_data(data: np.ndarray, max_val=None) -> np.ndarray:
    """
    Normalizes (rescales) the range of features to lie in the range [0, 1] and then vectorizes them, i.e.
    converts all the data instances to a 1-dimensional vector.

    Args:
    data: the input data to be normalized (shape: [num_samples x feature_dim_1 x feature_dim_2])
    max_val: the maximum value in the dataset. To be set as "None" for normalizing the training data and hence calculated. 
             If it is passed, then use that value instead of calculating it here. Only calculate it if it is None.

    Returns:
    normalized input data (shape: [num_samples, num_features])
    max_val (float): the maximum value in the dataset. 
    """
    ### Implement here

    if max_val is None:
        if len(data) == 0:
            raise ValueError('Could not find a max value due to empty data array ')
        max_val = np.max(data)

    data = [sample.flatten() for sample in data]
    data /= max_val
    return data, max_val


def euclidean_distance(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Computes the Euclidean distance between two arrays

    Args:
        A (np.ndarray): Numpy array of shape [num_samples_a x num_features]
        B (np.ndarray): Numpy array of shape [num_samples_b x num_features]

    Returns:
        np.ndarray: Numpy array of shape [num_samples_a x num_samples_b] where
                    each column contains the distance between one element in
                    matrix B and all elements in matrix A
    """

    if A.shape[1] != B.shape[1]:
        raise ValueError('Number of columns in A and B are different, cannot compute Euclidean distance')

    distances = scipy.spatial.distance.cdist(A, B, metric='euclidean')

    return distances


def cosine_distance(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Computes the cosine distance between two arrays

    Args:
        A (np.ndarray): Numpy array of shape [num_samples_a x num_features]
        B (np.ndarray): Numpy array of shape [num_samples_b x num_features]

    Returns:
        np.ndarray: Numpy array of shape [num_samples_a x num_samples_b] where
                    each column contains the cosine distance between one element in
                    matrix_b and all elements in matrix_a
    """
    # NOTE: Similar to the euclidean_distance function, you might want to use
    # SciPy's cdist function to avoid potential out-of-memory errors.
    distances = None

    ### Implement here

    if A.shape[1] != B.shape[1]:
        raise ValueError('Number of columns in A and B are different, cannot compute Euclidean distance')

    distances = scipy.spatial.distance.cdist(A, B, metric='cosine')

    return distances

def get_k_nearest_neighbors(distances: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Gets the k nearest labels based on the distances

    Args:
        distances (np.ndarray): Numpy array of shape num_train_samples x num_test_samples
                                containing the Euclidean distances
        labels (np.ndarray): Numpy array of shape [num_train_samples, ] containing
                                the training labels
        k (int): Number of nearest neighbours

    Returns:
        np.ndarray: Numpy array of shape [k x num_test_samples] containing the
                    training labels of the k nearest neighbours for each test sample
    """

    # Sort the distances in ascending and get the indices of the first "k" elements
    # HINT: You need to sort the distances in ascending order to get the indices
    # of the first "k" elements. BUT, you would not need to sort the entire array,
    # it would be enough to make sure that the "k"-th element is in the correct position!

    # NOTE: Since the matrix sizes are huge, it would be impractical to run any sort of a
    # loop to get the nearest labels. Think about how you can do it without using loops.

    neighbors = None

    ### Implement here

    smallest_indices = np.argpartition(distances, kth=k-1, axis=0)[:k]

    neighbors = labels[smallest_indices]

    return neighbors

def get_class_prediction(nearest_labels: np.ndarray) -> np.ndarray:
    """Gets the best prediction, i.e. the label class that occurs most frequently

    Args:
        nearest_labels (np.ndarray): Numpy array of shape [k x num_test_samples] obtained from the output of the get_k_neighbors function

    Returns:
        np.array: Numpy array of shape [num_test_samples] containing the best prediction for each test sample. If there are more than one most frequent labels, return the smallest one.
    """

    predicted = None

    ### Implement here

    predicted = scipy.stats.mode(nearest_labels, axis=0).mode

    return predicted

def knn_classifier(training_set: np.ndarray, training_labels: np.ndarray,
					test_set: np.ndarray, test_labels: np.ndarray, k: int) -> float:
    """
    Performs k-nearest neighbour classification

    Args:
    training_set (np.ndarray): Vectorized training images (shape: [num_train_samples x num_features])
    training_labels (np.ndarray): Training labels (shape: [num_train_samples, 1])
    test_set (np.ndarray): Vectorized test images (shape: [num_test_samples x num_features])
    test_labels (np.ndarray): Test labels (shape: [num_test_samples, 1])
    k (int): number of nearest neighbours

    Returns:
    accuracy (float): the accuracy in %
    """

    dists = euclidean_distance(A=training_set, B=test_set)

    nearest_labels = get_k_nearest_neighbors(distances=dists, labels=training_labels, k=k)

    # from the K nearest labels above choose the label classes that occurs most frequently
    predictions = get_class_prediction(nearest_labels)

    # calculate and return accuracy of the predicitions
    accuracy = (np.equal(predictions, test_labels).sum())/len(test_set) * 100.0

    return accuracy

def knn_classifier_cosine(training_set: np.ndarray, training_labels: np.ndarray,
					test_set: np.ndarray, test_labels: np.ndarray, k: int) -> float:
    """
    Performs k-nearest neighbour classification with cosine distance

    Args:
    training_set (np.ndarray): Vectorized training images (shape: [num_train_samples x num_features])
    training_labels (np.ndarray): Training labels (shape: [num_train_samples, 1])
    test_set (np.ndarray): Vectorized test images (shape: [num_test_samples x num_features])
    test_labels (np.ndarray): Test labels (shape: [num_test_samples, 1])
    k (int): number of nearest neighbours

    Returns:
    accuracy (float): the accuracy in %
    """

    dists = cosine_distance(A=training_set, B=test_set)

    nearest_labels = get_k_nearest_neighbors(distances=dists, labels=training_labels, k=k)

    # from the K nearest labels above choose the label classes that occurs most frequently
    predictions = get_class_prediction(nearest_labels)

    # calculate and return accuracy of the predicitions
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
    # dictionary to store the k values as keys and the validation accuracies as the values
    val_accuracy_per_k = {}

    for k in [2, 3, 4, 5, 6, 7, 8]:
        print(f"Calculating validation accuracy for k={k}")
        val_accuracy_per_k[k] = knn_classifier(train_set, train_labels, val_set, val_labels, k)
        print(f"Validation accuracy of {val_accuracy_per_k[k]} % for k={k}")

    best_k = max(val_accuracy_per_k, key=val_accuracy_per_k.get)
    print(f"Best validation accuracy of {val_accuracy_per_k[best_k]} % for k={best_k}")

    print ("Running on the test set...")
    test_accuracy = knn_classifier(train_set, train_labels, test_set, test_labels, k=best_k)
    print(test_accuracy)


    k_values = list(val_accuracy_per_k.keys())
    accuracies = list(val_accuracy_per_k.values())

    plt.figure(figsize=(8, 6))
    plt.plot(k_values, accuracies, marker='o')
    plt.title('Validation Accuracy according to k using Eucledian distance')
    plt.xlabel('k')
    plt.ylabel('Validation Accuracy (%)')
    plt.grid(True)
    plt.show()


    # Uncomment the following part of the code to run knn with cosine distance
    val_accuracy_per_k_cosine = {}

    for k in [2,3]:
        print(f"Calculating validation accuracy for k={k} with cosine distance")
        val_accuracy_per_k_cosine[k] = knn_classifier_cosine(train_set, train_labels, val_set, val_labels, k)
        print(f"Validation accuracy of {val_accuracy_per_k_cosine[k]} % for k={k} with cosine distance")

    best_k_cosine = max(val_accuracy_per_k_cosine, key=val_accuracy_per_k_cosine.get)
    print(f"Best validation accuracy of {val_accuracy_per_k_cosine[best_k_cosine]} % for k={best_k_cosine} with cosine distance")

    print ("Running on the test set for cosine distance...")
    test_accuracy = knn_classifier_cosine(train_set, train_labels, test_set, test_labels, k=best_k_cosine)
    print(test_accuracy)

    k_cos_values = list(val_accuracy_per_k_cosine.keys())
    accuracies_cos = list(val_accuracy_per_k_cosine.values())

    plt.figure(figsize=(8, 6))
    plt.plot(k_cos_values, accuracies_cos, marker='o')
    plt.title('Validation Accuracy according to k using Cosine distance')
    plt.xlabel('k')
    plt.ylabel('Validation Accuracy (%)')
    plt.grid(True)
    plt.show()
