import os
import numpy as np

from typing import List, Tuple, Union, Dict, Callable
from math import sqrt
from numpy.random import rand
import torch

## SET GLOBAL SEED
## Do not modify this for reproducibility
np.random.seed(33)

"""
The following class `Layer` is a super-class from which subsequent layer, activation, and loss classes will be inherited.
Nothing needs to be changed here, but it's good to get familiar with the stucture and the instuctions.
"""


class Layer:
    """
    Base class for all layers.
    """

    def __init__(self):
        """
        Initialize layer parameters (if any) and auxiliary data.
        Parameters:
            input_size (int, optional): Size of the input to the layer.
            output_size (int, optional): Size of the output from the layer.
        """
        self.weights = None
        self.bias = None

    def init_weights(self):
        """
        Initialize the weights of the layer, if applicable.
        """
        pass

    def forward(self, input):
        """
        Forward pass of the layer.
        Parameter:
            input: input data
        Returns:
            output: output of the layer
        """
        pass

    def backward(self, output_grad):
        """
        Backward pass of the layer.
        Parameter:
            output_grad: gradient of the output of the layer (dy)
        Returns:
            input_grad: gradient of the input of the layer (dx)
        """
        pass

    def update(self, learning_rate):
        """
        Update the layer parameters, if applicable.
        Parameter:
            learning_rate: learning rate used for updating
        """
        pass

    def __call__(self, input, *args, **kwargs):
        """
        Call the forward pass of the layer.
        Parameter:
            input: input data
        Returns:
            output: output of the layer
        """
        return self.forward(input, *args, **kwargs)


"""### Q1.1: Layer Class
Follow the instuctions in the comments to implement a fully-connected layer (`Dense`) capable
of working with any generic input and output sizes. It should define its weight and bias matrices inside.
"""


class Dense(Layer):
    """
    Fully connected layer.
    """

    def __init__(
        self,
        input_size,
        output_size,
    ):
        """
        Initialize the layer.
        Parameters:
            input_size (int): input size of the layer
            output_size (int): output size of the layer
            weights (np.ndarray): weights of the layer
            bias (np.ndarray): bias of the layer
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.init_weights()

    def init_weights(self, weights=None, bias=None):
        """
        Initializes the weights of the layer.
        By default, the weights and biases are initialized using the
        uniform Xavier initialization.
        Parameters:
            weights (np.ndarray): weights of the layer, shape: (self.input_size, self.output_size)
            bias (np.ndarray): bias of the layer, shape: (1, self.output_size)
        """
        ## RESET GLOBAL SEED
        ## Do not modify this for reproducibility
        np.random.seed(33)

        # BEGIN SOLUTION

        # TODO: Check dimensions
        n = 6
        lower, upper = -(sqrt(n / (self.input_size + self.output_size))), (
            sqrt(n / (self.input_size + self.output_size))
        )

        numbers = rand(1000)

        self.weights = np.random.uniform(
            low=lower, high=upper, size=(self.input_size, self.output_size)
        )

        self.bias = np.zeros(self.output_size)
        # END SOLUTION

    def forward(self, x):
        """
        Forward pass of the layer.
        Parameters:
            x (np.ndarray): input of the layer, shape: (batch_size, self.input_size)
        Returns:
            output (np.ndarray): output of the layer, shape: (batch_size, self.output_size)
        """
        # BEGIN SOLUTION
        output = np.dot(x, self.weights) + self.bias
        self.input = torch.tensor(x, dtype=torch.float32)
        self.output = output
        return output
        # END SOLUTION

    def backward(self, output_grad):
        """
        Backward pass of the layer.
        Parameters:
            output_grad (np.ndarray): gradient of the output of the layer (dy), shape: (batch_size, self.output_size)
        Returns:
            input_grad (np.ndarray): gradient of the input of the layer (dx), shape: (batch_size, self.input_size)
        """
        # BEGIN SOLUTION

        batch_size = self.input.shape[0]

        self.weights_grad = np.dot(self.input.T, output_grad) / batch_size
        self.bias_grad = np.mean(output_grad, axis=0)
        input_grad = np.dot(output_grad, self.weights.T)

        return input_grad
        # END SOLUTION

    def update(self, learning_rate):
        """
        Update the layer parameters. Normally, this is done by using the
        gradients computed in the backward pass; therefore, backward() must
        be called before update().
        This function implements SGD (stochastic gradient descent)
        Parameter:
            learning_rate (float): learning rate used for updating
        """
        # assumes self.backward() function has been called before
        assert hasattr(self, "weights_grad"), "must compute gradient of weights before"
        assert hasattr(self, "bias_grad"), "must compute gradient of bias before"
        # BEGIN SOLUTION
        self.weights -= learning_rate * self.weights_grad
        self.bias -= learning_rate * self.bias_grad
        # END SOLUTION


"""
## **Question 1.2: Activation and Loss Layers (30 points)**
"""


class SoftmaxLayer(Layer):
    """
    Softmax layer.
    """

    def forward(self, x):
        """
        Forward pass of the layer.
        The output's sum along the second axis should be 1.
        Parameter:
            x (np.ndarray): input of the layer, shape: (batch_size, input_size)
        Returns:
            output (np.ndarray): output of the layer, shape: (batch_size, input_size)
        """
        # BEGIN SOLUTION
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        self.output = output
        return output
        # END SOLUTION

    def backward(self, output_grad):
        """
        Backward pass of the layer.
        Parameter:
            output_grad (np.ndarray): gradient of the output of the layer (dy), shape: (batch_size, input_size)
        Returns:
            input_grad (np.ndarray): gradient of the input of the layer (dx), shape: (batch_size, input_size)
        """
        # BEGIN SOLUTION
        batch_size, num_classes = self.output.shape
        input_grad = np.zeros_like(output_grad)

        for i in range(batch_size):
            s = self.output[i].reshape(-1, 1)
            jacobian = np.diagflat(s) - np.dot(s, s.T)
            input_grad[i] = np.dot(jacobian, output_grad[i])

        return input_grad
        # END SOLUTION


class TanhLayer(Layer):
    """
    Tanh layer.
    """

    def forward(self, x):
        """
        Forward pass of the layer.
        Parameter:
            x (np.ndarray): input of the layer, shape: (batch_size, input_size)
        Returns:
            output (np.ndarray): output of the layer, shape: (batch_size, input_size)
        """
        # BEGIN SOLUTION
        self.output = np.tanh(x)
        return self.output

        # END SOLUTION

    def backward(self, output_grad):
        """
        Backward pass of the layer.
        Parameter:
            output_grad (np.ndarray): gradient of the output of the layer (dy), shape: (batch_size, input_size)
        Returns:
            input_grad (np.ndarray): gradient of the input of the layer (dx), shape: (batch_size, input_size)
        """
        # BEGIN SOLUTION
        input_grad = output_grad * (1 - self.output**2)
        return input_grad
        # END SOLUTION


class ReLULayer(Layer):
    """
    ReLU layer.
    """

    def forward(self, x):
        """
        Forward pass of the layer.
        Parameter:
            x (np.ndarray): input of the layer, shape: (batch_size, input_size)
        Returns:
            output (np.ndarray): output of the layer, shape: (batch_size, input_size)
        """
        # BEGIN SOLUTION
        self.inputs = x
        return np.maximum(0, x)
        # END SOLUTION

    def backward(self, output_grad):
        """
        Backward pass of the layer.
        Parameter:
            output_grad (np.ndarray): gradient of the output of the layer (dy), shape: (batch_size, input_size)
        Returns:
            input_grad (np.ndarray): gradient of the input of the layer (dx), shape: (batch_size, input_size)
        """
        # BEGIN SOLUTION

        input_grad = output_grad * (self.inputs > 0).astype(float)
        return input_grad
        # END SOLUTION


"""
## Q1.3: Cross-Entropy Loss Layer

The forward pass again receives the predicted class probabilities (output from a previous activation layer)
and the ground truth labels (target). It computes the cross-entropy loss using the predicted probabilities and the ground truth labels.
The loss function measures the dissimilarity between the predicted probabilities and the actual labels.

In the backward pass, you need to compute the gradient of the loss with respect to the predicted probabilities. This gradient will be used to update the weights and biases of the preceding layers during backpropagation.

The equation for the cross-entropy loss is given by:
$$
\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} y_i \log(p_i)
$$
where $y_i$ is the ground truth label, $p_i$ is the predicted probability for class $i$, and $N$ is the batch size.

Remember to handle numerical stability, it is a good practice to add a small value (e.g. $10^{-10}$) to the predicted probabilities before taking the logarithm.
"""


class CrossEntropyLossLayer(Layer):
    """
    Cross entropy loss layer.
    """

    def forward(self, prediction, target):
        """
        Forward pass of the layer.
        Note that prediction input is assumed to be a probability distribution (e.g., softmax output).
        Parameters:
            prediction (np.ndarray): prediction of the model, shape: (batch_size, num_classes)
            target (np.ndarray): target, shape: (batch_size,)
        Returns:
            output (float): cross entropy loss, averaged over the batch
        """
        # BEGIN SOLUTION

        exp_prediction = np.exp(prediction - np.max(prediction, axis=1, keepdims=True))
        softmax_prediction = exp_prediction / np.sum(
            exp_prediction, axis=1, keepdims=True
        )

        self.prediction = softmax_prediction
        self.target = target

        epsilon = 1e-10
        batch_size = prediction.shape[0]
        log_likelihood = -np.log(
            softmax_prediction[np.arange(batch_size), target] + epsilon
        )
        loss = np.sum(log_likelihood) / batch_size
        return loss

    def backward(self, output_grad=1.0):
        """
        Backward pass of the layer.
        Parameter:
            output_grad (float): gradient of the output of the layer (dy)
        Returns:
            input_grad (np.ndarray): gradient of the input of the layer (dx), shape: (batch_size, num_classes)
        """
        # BEGIN SOLUTION
        batch_size = self.prediction.shape[0]
        grad = self.prediction
        grad[np.arange(batch_size), self.target] -= 1
        grad /= batch_size

        return grad * output_grad
        # END SOLUTION
