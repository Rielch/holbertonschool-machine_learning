#!/usr/bin/env python3
"""function that performs a convolution on grayscale images"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """Performs a convolution on grayscale images"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride
    if padding == 'valid':
        pad_h, pad_w = 0, 0
    elif padding == 'same':
        pad_h = (((h - 1) * sh + kh - h) // 2) + 1
        pad_w = (((w - 1) * sw + kw - w) // 2) + 1
    else:
        pad_h, pad_w = padding
    out_h = ((h - kh + 2 * pad_h) // sh) + 1
    out_w = ((w - kw + 2 * pad_w) // sw) + 1
    padded = np.pad(images, ((0,), (pad_h,), (pad_w,)), 'constant')
    conv = np.zeros(shape=(m, out_h, out_w))
    for i in range(0, (out_h * out_w)):
        row = i // out_w
        col = i % out_w
        conv[:, row, col] = (
                padded[:, row * sh:kh + row * sh, col * sw:kw + col * sw]
                * kernel).sum(axis=(1, 2))
    return conv
