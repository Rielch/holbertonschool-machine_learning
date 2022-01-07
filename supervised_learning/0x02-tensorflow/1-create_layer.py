#!/usr/bin/env python3
"""Function that returns the tensor output of the layer"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """Returns the tensor output of the layer"""
    layer = tf.layers.Dense(
        n, activation,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
            mode="FAN_AVG"),
        name="Layer")
    return layer(prev)
