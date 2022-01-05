#!/usr/bin/env python3
"""Define a neuron for binary classification"""
import numpy as np


class Neuron:
    """Neuron that performs binary classification"""

    def __init__(self, nx):
        """Initializates the neuron"""

        if type(nx) is not int:
            raise TypeError('nx must be an integer')

        if nx < 1:
            raise ValueError('nx must be a positive integer')

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Gets the value of W"""
        return self.__W

    @property
    def b(self):
        """Gets the value of b"""
        return self.__b

    @property
    def A(self):
        """Gets the value of A"""
        return self.__A

    def forward_prop(self, X):
        """Calculates the forward propagation"""
        forward_prop = np.matmul(self.__W, X) + self.__b
        sigmoid = 1 / (1 + (np.e**(-forward_prop)))
        self.__A = sigmoid
        return self.__A

    def cost(self, Y, A):
        """Calculates the cost of the neuron"""
        m = Y.shape[1]
        a = 1.0000001 - A
        y = 1 - Y
        cost_ = -(1 / m) * np.sum(Y * np.log(A) + y * np.log(a))
        return cost_

    def evaluate(self, X, Y):
        """Evaluates the network  predictions"""
        self.forward_prop(X)
        prediction = np.where(self.__A >= 0.5, 1, 0)
        cost = self.cost(Y, self.__A)
        return (prediction, cost)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculates one pass of gradient descent on the neuron"""
        m = X.shape[1]
        z = A - Y
        w = np.matmul(X, z.T) / m
        b = np.sum(z) / m
        self.__W = self.__W - (alpha * w).T
        self.__b = self.__b - (alpha * b).T

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose = True,
              graph=True, step=100):
        """Trains the neuron"""
        import matplotlib.pyplot as plt

        if type(iterations) is not int:
            raise TypeError('iterations must be an integer')
        if iterations < 0:
            raise ValueError('iterations must be a positive integer')
        if type(alpha) is not float:
            raise TypeError('alpha must be a float')
        if alpha < 0:
            raise ValueError('alpha must be positive')
        cost_list = []
        for i in range(iterations + 1):
            cost = self.cost(Y, self.__A)
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A, alpha)
            if verbose and i % step == 0:
                print('Cost after {} iterations: {}'.format(i,
                      self.cost(Y, self.__A)))
                if i < iterations:
                    cost_list.append(cost)
        if graph:
            x = np.arange(0, iterations, step)
            y = cost_list
            plt.plot(x, y)
            plt.title('Training Cost')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.show()
        return self.evaluate(X, Y)
