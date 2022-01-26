#!/usr/bin/env python3
"""Function that builds a neural network with the Keras library"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Builds a neural network with the Keras library"""
    inputs = K.Input(shape=(nx,))
    x = inputs
    for i in range(len(layers)):
        if i != 0:
            x = K.layers.Dropout(1 - keep_prob)(x)
        x = K.layers.Dense(
            layers[i],
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha))(x)
    return K.Model(inputs, x)
