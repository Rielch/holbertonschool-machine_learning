#!/usr/bin/env python3
"""Function that builds a neural
 network with the keras library"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Builds a neural network with the keras library"""
    model = K.Sequential()
    regularizer = K.regularizers.L2(l2=lambtha)
    model.add(K.Input(shape=(nx,)))
    for i, l in enumerate(layers):
        model.add(K.layers.Dense(l, activation=activations[i],
                                 kernel_regularizer=regularizer))
        if i < len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))

    return model
