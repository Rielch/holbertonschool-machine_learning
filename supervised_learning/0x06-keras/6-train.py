#!/usr/bin/env python3
"""Function that trains a model using mini-batch gradient descent"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """Ttrains a model using mini-batch gradient descent"""
    if early_stopping:
        callback = K.callbacks.EarlyStopping(monitro='var_loss',
                                             patience=patience)
    else:
        callback = None
    history = network.fit(x=data, y=labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=validation_data,
                          callbacks=[callback],
                          verbose=verbose,
                          shuffle=shuffle)
    return history
