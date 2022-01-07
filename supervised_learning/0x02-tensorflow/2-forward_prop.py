#!/usr/bin/env python3
"""Function that creates the forward
 propagation graph for the neural network"""
import tensorflow as tf


def forward_prop(x, layer_sizes=[], activations=[]):
    """Creates the forward propagation
    graph for the neural network"""
    create_layer = __import__('1-create_layer').create_layer
    for i in range(len(layer_sizes)):
        layer = create_layer(x, layer_sizes[i], activations[i])
        x = layer
    return x
