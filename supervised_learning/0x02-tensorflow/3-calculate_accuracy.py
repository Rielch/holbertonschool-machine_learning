#!/usr/bin/env python3
"""Function that calculates the accuracy of a prediction"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """Calculates the accuracy of a prediction"""
    accuracy = tf.math.reduce_mean(tf.cast(tf.equal(
        tf.argmax(y, axis=-1), tf.argmax(y_pred, axis=-1)),
                                           tf.float32))
    return accuracy
