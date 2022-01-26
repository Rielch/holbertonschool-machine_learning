#!/usr/bin/env python3
"""Functions that loads and saves the configuration"""
import tensorflow.keras as K


def save_config(network, filename):
    """Saves the configuration"""
    with open(filename, "w") as f:
        f.write(network.to_json())


def load_config(filename):
    """Loads the configuration"""
    with open(filename, "r") as f:
        conf_json = f.read()
    return K.models.model_from_json(conf_json)
