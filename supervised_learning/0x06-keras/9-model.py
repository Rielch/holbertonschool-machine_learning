#!/usr/bin/env python3
"""Functions that load and save the model"""
import tensorflow.keras as K


def save_model(network, filename):
    """Saves a model"""
    network.save(filename)


def load_model(filename):
    """Loads a model"""
    return K.models.load_model(filename)
