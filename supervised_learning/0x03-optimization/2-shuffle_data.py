#!/usr/bin/env python3
"""Function that shuffles the data
 points in two matrices the same way"""
import numpy as np


def shuffle_data(X, Y):
    """Shuffles the data points in
    two matrices the same way"""
    shuffle = np.random.permutation(X.shape[0])
    x_shuffle = X[shuffle]
    y_shuffle = Y[shuffle]
    return x_shuffle, y_shuffle
