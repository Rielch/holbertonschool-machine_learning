#!/usr/bin/env python3
"""Function that performs backward propagation over a
convolutional layer of a neural network"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """Performs backward propagation over a
    convolutional layer of a neural network"""
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    if padding == 'same':
        pad_h = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pad_w = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:
        pad_h, pad_w = 0, 0
    A_prevp = np.pad(
        A_prev, ((0,), (pad_h,), (pad_w,), (0,)), 'constant')
    dW = np.zeros(shape=(kh, kw, c_prev, c_new))
    dA_prev = np.zeros_like(A_prevp)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    for im in range(m):
        for i in range(0, (h_new * w_new)):
            row = i // w_new
            col = i % w_new
            for kernel in range(c_new):
                dW[:, :, :, kernel] += A_prevp[
                    im, row * sh:kh + row * sh, col * sw:kw + col * sw, :
                ] * dZ[im, row, col, kernel]
                dA_prev[im, row * sh:kh + row * sh, col * sw:kw + col *
                        sw, :] += dZ[im, row, col, kernel] * W[:, :, :, kernel]
    dA_prev = dA_prev[:, pad_h:h_prev + pad_h, pad_w:pad_w + w_prev, :]
    return dA_prev, dW, db
