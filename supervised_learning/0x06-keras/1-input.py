#!/usr/bin/env python3
"""Function that builds a neural network with the Keras library"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Builds a neural network with the Keras library"""
    regularizer = K.regularizers.L2(l2=lambtha)
    input_layer = K.Input(shape=(nx,))
    x = input_layer
    for i in range(len(layers)):
        if i != 0:
            x = K.layer.Dropout(1 - keep_prob)(x)
        x = K.layer.Dense(layers[i],
                          activation=activations[i],
                          kernel_regularizer=regularizer)(x)
        return K.Model(input_layer, x)
