#!/usr/bin/env python3
"""Function that sets up Adam optimization for a keras
model with categorical crossentropy loss and accuracy metrics"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """Sets up Adam optimization for a keras model with
    categorical crossentropy loss and accuracy metrics"""
    model = network
    optimizers = K.optimizers.Adam(alpha, beta1, beta2)
    model.compile(loss='categorical_crossentropy',
                  metrics=['accuracy'],
                  optimizer=optimizers)
