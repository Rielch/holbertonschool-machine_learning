#!/usr/bin/env python3
"""Function that returns the tensor output of the layer"""
import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """Returns the tensor output of the layer"""
    new_layer = tf.layers.Dense(
        n, activation,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            mode='fan_avg'),
        name="layer")
    return new_layer(prev)
