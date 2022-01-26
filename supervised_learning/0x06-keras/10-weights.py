#!/usr/bin/env python3
"""Functions that loads and saves the weights"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """Saves the weights"""
    network.save_weights(filename, save_format=save_format)


def load_weights(network, filename):
    """Loads the weights"""
    network.load_weights(filename)
