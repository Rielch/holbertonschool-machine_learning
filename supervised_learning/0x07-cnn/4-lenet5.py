#!/usr/bin/env python3
"""Function that builds a modified version of
 the LeNet-5 architecture using tensorflow"""
import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """Builds a modified version of the
    LeNet-5 architecture using tensorflow"""
    kernel = tf.keras.initializers.VarianceScaling(scale=2.0)
    L1_conv = tf.layers.Conv2D(inputs=x,
                               filters=6,
                               kernel_size=(5, 5),
                               kernel_initializer=kernel,
                               padding="same",
                               activation='relu')
    L2_pool = tf.layers.MaxPooling2D(inputs=L1_conv,
                                      pool_size=(2, 2),
                                      strides=(2, 2))
    L3_conv = tf.layers.Conv2D(inputs=L2_pool,
                               filters=16,
                               kernel_size=(5, 5),
                               kernel_initializer=kernel,
                               activation='relu')
    L4_pool = tf.layers.MaxPooling2D(inputs=L3_conv,
                                      pool_size=(2, 2),
                                      strides=(2, 2))
    L4_flat = tf.layers.flatten(L4_pool)
    L5_fc = tf.layers.Dense(inputs=L4_flat,
                            units=120,
                            kernel_initializer=kernel,
                            activation='relu')
    L6_fc = tf.layers.Dense(inputs=L5_fc,
                            units=84,
                            kernel_initializer=kernel,
                            activation='relu')
    L7_fc = tf.layers.Dense(inputs=L6_fc,
                            units=10,
                            kernel_initializer=kernel,)
    Y_pred = tf.nn.softmax(L7_fc)
    loss = tf.losses.softmax_cross_entropy(y, L7_fc)
    Train_op = tf.train.AdamOptimizer().minimize(loss)
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(L7_fc, axis=1),
                                          tf.argmax(y, axis=1)), "float32"))
    return Y_pred, Train_op, loss, acc
