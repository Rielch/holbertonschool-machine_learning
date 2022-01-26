#!/usr/bin/env python3
"""Function that trains a model using mini-batch gradient descent"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, verbose=True, shuffle=False):
    """Ttrains a model using mini-batch gradient descent"""
    def scheduler(epoch):
        return alpha / (1 + (decay_rate * epoch))
    callback = []
    if early_stopping:
        callback.append(K.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=patience))
    if learning_rate_decay and validation_data:
        callback.append(K.callbacks.LearningRateScheduler(
            scheduler, verbose=1))
    history = network.fit(x=data, y=labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=validation_data,
                          callbacks=callback,
                          verbose=verbose,
                          shuffle=shuffle)
    return history
