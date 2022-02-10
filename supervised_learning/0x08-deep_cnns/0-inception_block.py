#!/usr/bin/env python3
"""Function that builds an inception block"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """Builds an inception block"""
    F1 = filters[0]
    F3R = filters[1]
    F3 = filters[2]
    F5R = filters[3]
    F5 = filtes[4]
    FPP = filters[5]
    c1x1 = K.layers.Conv2D(F1, (1, 1), padding='same',
                           activation='relu')(A_prev)
    c3x3 = K.layers.Conv2D(F3R, (1, 1), padding='same',
                           activation='relu')(A_prev)
    c3x3 = K.layers.Conv2D(F3, (3, 3), padding='same', activation='relu')(c3x3)
    c5x5 = K.layers.Conv2D(F5R, (1, 1), padding='same',
                           activation='relu')(A_prev)
    c5x5 = K.layers.Conv2D(F5, (5, 5), padding='same', activation='relu')(c5x5)
    poolPr = K.layers.MaxPool2D((3, 3), strides=(1, 1), padding='same')(A_prev)
    poolPr = K.layers.Conv2D(FPP, (1, 1), padding='same',
                             activation='relu')(poolPr)
    concOutput = K.layers.concatenate([c1x1, c3x3, c5x5, poolPr], axis=3)
    return concOutput
