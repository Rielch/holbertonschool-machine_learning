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

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = Y.shape[1]
        a = 1.0000001 - A
        y = 1 - Y
        cost_ = -(1 / m) * np.sum(Y * np.log(A) + y * np.log(a))
        return cost_

    def evaluate(self, X, Y):
        """Evaluates the neural network’s predictions"""
        self.forward_prop(X)
        prediction = np.where(self.__A2 >= 0.5, 1, 0)
        cost = self.cost(Y, self.__A2)
        return (prediction, cost)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network"""
        m = X.shape[1]
        z2 = A2 - Y
        w2 = np.matmul(z2, A1.T) / m
        b2 = np.sum(z2, axis=1, keepdims=True) / m
        z1 = np.matmul(self.__W2.T, z2) * (A1 * (1 - A1))
        w1 = np.matmul(z1, X.T) / m
        b1 = np.sum(z1, axis=1, keepdims=True) / m
        self.__W2 -= alpha * w2
        self.__b2 -= alpha * b2
        self.__W1 -= alpha * w1
        self.__b1 -= alpha * b1

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """Trains the neural network"""
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
            cost = self.cost(Y, self.__A2)
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)
            if verbose and i % step == 0:
                print('Cost after {} iterations: {}'.format(i, cost))
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
