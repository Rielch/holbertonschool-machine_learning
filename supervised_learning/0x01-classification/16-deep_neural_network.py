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
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        for i in range(self.L):
            if type(layers[i]) is not int or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")
            if i == 0:
                a = nx
            else:
                a = layers[i - 1]
            W = np.random.randn(layers[i], a) * np.sqrt(2 / a)
            self.weights["W" + str(i + 1)] = W
            self.weights["b" + str(i + 1)] = np.zeros((layers[i], 1))
