#!/usr/bin/env python3
"""Function that performs forward propagation over a
convolutional layer of a neural network"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Performs forward propagation over a
    convolutional layer of a neural network"""
    m, h, w, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    out_h = ((h - kh) // sh) + 1
    out_w = ((w - kw) // sw) + 1
    pool = np.zeros(shape=(m, out_h, out_w, c))
    for i in range(0, (out_h * out_w)):
        row = i // out_w
        col = i % out_w
        if mode == 'max':
            pool[:, row, col, :] = (
                A_prev[:, row * sh:kh + row * sh, col * sw:kw + col * sw, :]
            ).max(axis=(1, 2))
        elif mode == 'avg':
            pool[:, row, col, :] = np.average(
                A_prev[:, row * sh:kh + row * sh, col * sw:kw + col * sw, :],
                axis=(1, 2))
    return pool
