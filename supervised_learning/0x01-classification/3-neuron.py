#!/usr/bin/env python3
"""Define a neuron for binary classification"""
import numpy as np


class Neuron:
    """Neuron that performs binary classification"""

    def __init__(self, nx):
        """Initializes the neuron"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter for the weights of the neuron"""
        return self.__W

    @property
    def b(self):
        """Getter for the bias of the neuron"""
        return self.__b

    @property
    def A(self):
        """Getter for the activation of the neuron"""
        return self.__A

    def forward_prop(self, X):
        """Calculates the forward propagation of a neuron"""
        forward = np.matmul(self.__W, X) + self.__b
        sigmoid = 1 / (1 + (np.e ** (-forward)))
        self.__A = sigmoid
        return self.__A

    def cost(self, Y, A):
        """Calcualtes the cost of the model"""
        m = Y.shape[1]
        a = 1.0000001 - A
        y = 1 - a
        cost = -(1 / m) * np.sum(Y * np.log(A) + y * np.log(a))
        return cost
