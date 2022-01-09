#!/usr/bin/env python3
"""Defines a Neural Network with one hidden layer"""
import numpy as np


class NeuralNetwork:
    """Defines a Neural Network"""

    def __init__(self, nx, nodes):
        """Initializes the Neural Network"""

        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(nodes) is not int:
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Gets the value of W1"""
        return self.__W1

    @property
    def W2(self):
        """Gets the value of W2"""
        return self.__W2

    @property
    def b1(self):
        """Gets the value of b1"""
        return self.__b1

    @property
    def b2(self):
        """Gets the value of b2"""
        return self.__b2

    @property
    def A1(self):
        """Gets the value of A1"""
        return self.__A1

    @property
    def A2(self):
        """Gets the value of A2"""
        return self.__A2

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""
        forward_prop1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + (np.e**-forward_prop1))
        forward_prop2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + (np.e**-forward_prop2))
        return self.__A1, self.__A2
