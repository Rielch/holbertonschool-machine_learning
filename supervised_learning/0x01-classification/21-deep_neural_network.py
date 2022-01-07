#!/usr/bin/env python3
"""Defines a deep learning network for binary classification"""
import numpy as np


class DeepNeuralNetwork:
    """Deep learning network for binary classification"""

    def __init__(self, nx, layers):
        """Initializates the deep learning network"""
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        elif nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(layers) is not list or len(layers) == 0:
            raise TypeError('layers must be a list of positive integers')
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for i in range(self.L):
            if type(layers[i]) is not int or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")
            if i == 0:
                a = nx
            else:
                a = layers[i - 1]
            W = np.random.randn(layers[i], a) * np.sqrt(2 / a)
            self.__weights["W" + str(i + 1)] = W
            self.__weights["b" + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """Gets the value of L"""
        return self.__L

    @property
    def cache(self):
        """Gets the value of cache"""
        return self.__cache

    @property
    def weights(self):
        """Gets the values of weights"""
        return self.__weights

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""
        self.__cache["A0"] = X
        for i in range(self.__L):
            self.__cache["A" + str(i + 1)] = self.sigmoid(np.dot(
                self.__weights["W" + str(i + 1)],
                self.__cache["A" + str(i)]) + self.__weights["b" +
                                                             str(i + 1)])
        return self.__cache["A" + str(self.__L)], self.__cache

    @staticmethod
    def sigmoid(x):
        """Performs sigmoid in a number"""
        return 1.0 / (1.0 + np.exp(-x))

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = np.shape(Y)[1]
        cost_ = -1 / m * np.sum(Y * np.log(A) + (1 - Y) *
                                (np.log(1.0000001 - A)))
        return cost_

    def evaluate(self, X, Y):
        """Calculates the cost of the model using logistic regression"""
        A = self.forwardprop(X)[0]
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, self.cost(Y, A)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network"""
        m = shape(Y)[1]
        grad = cache["A" + str(self.__L)] - Y
        for i in range(self.__L, 0, -1):
            grad1 = np.matmul(grad, cache["A" + str(i - 1)].T) / m
            grad2 = np.sum(grad, axis=1, keepdims=True) / m
            grad = np.matmul(self.__waights["W" + str(i)].T, grad) * (
                cache["A" + str(i - 1)] * (1 - cache["A" + str(i - 1)]))
            self.__weights["W" + str(i)] -= alpha * grad1
            self.__weights["b" + str(i)] -= alpha * grad2
