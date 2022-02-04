#!/usr/bin/env python3
"""Function that performs backward propagation over a
pooling layer of a neural network"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Performs backward propagation over a
    pooling layer of a neural network"""
    m, h_new, w_new, c_new = dA.shape
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    dA_prev = np.zeros_like(A_prev)
    for im in range(m):
        for i in range(0, (h_new * w_new)):
            row = i // w_new
            col = i % w_new
            for kernel in range(c_new):
                slice_A = A_prev[im, row * sh:kh + row *
                                 sh, col * sw:kw + col * sw, kernel]
                slice_dA = dA_prev[im, row * sh:kh + row *
                                   sh, col * sw:kw + col * sw, kernel]
                if mode == 'max':
                    mask = (slice_A == np.max(slice_A))
                    slice_dA += dA[im, row, col, kernel] * mask
                if mode == 'avg':
                    slice_dA += dA[im, row, col, kernel] / (kh * kw)
    return dA_prev
