#!/usr/bin/env python3
"""Python script that trains a convolutional
neural network to classify the CIFAR 10 dataset"""
import os
import tensorflow.keras as K
import numpy as np


def preprocess_data(X, Y):
    """Function that preprocess the data of the model"""
    X_p = X / 255.0
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


