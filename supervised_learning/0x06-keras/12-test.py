#!/usr/bin/env python3
"""Function that tests a neural network"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """Ttests a neural network"""
    return network.evaluate(data, labels, verbose=verbose)
