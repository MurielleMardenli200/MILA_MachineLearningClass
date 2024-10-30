# Do not modify this file
from matplotlib import pyplot
import numpy as np

seed = 42
np.random.seed(seed)
rng = np.random.RandomState(seed)

from dataset import extract_training_samples, extract_test_samples

x_train, y_train = extract_training_samples('letters')
x_test, y_test = extract_test_samples('letters')

# Only keep the first 15 letters. Note that the numbering starts from 1 and not 0 for the labels.
train_idx = np.where(y_train<=15)
x_train = x_train[train_idx]
y_train = y_train[train_idx]
test_idx = np.where(y_test<=15)
x_test = x_test[test_idx]
y_test = y_test[test_idx]
y_train = y_train - 1
y_test = y_test - 1

# Shuffle the training set
permuted = np.random.permutation(len(x_train))
x_train, y_train = x_train[permuted], y_train[permuted]

print(f"Number of images for training: {x_train.shape[0]}")
print(f"Number of images for testing: {x_test.shape[0]}")
print(f"Size of  MNIST images: {x_train[0].shape}")

if __name__ == "__main__":
    # Use this to plot some instances
    for i in range(9):
        pyplot.subplot(330 + 1 + i)
        pyplot.imshow(x_train[i], cmap=pyplot.get_cmap('gray'))
        pyplot.show()

    print(f"Labels: {[chr(65+i) for i in y_train[:9]]}")