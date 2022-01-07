#!/usr/bin/env python3
"""Function that returns two placeholders,
 x and y, for the neural network"""
import tensorflow as tf


def create_placeholders(nx, classes):
    """returns two placeholders, x and y,
    for the neural network"""
    x = tf.placeholder("float", [None, nx], "x")
    y = tf.placeholder("float", [None, classes], "y")
    return x, y
