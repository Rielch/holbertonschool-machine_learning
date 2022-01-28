#!/usr/bin/env python3
"""function that performs a convolution on
grayscale images with custom padding"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """Performs a convolution on grayscale
    images with custom padding"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    pad_h, pad_w = padding
    out_h = (h + (pad_h * 2)) - kh + 1
    out_w = (w + (pad_w * 2)) - kw + 1
    conv = np.zeros(shape=(m, out_h, out_w))
    if pad_h == 0 and pad_w == 0:
        padded = images
    else:
        padded = np.zeros(shape=(m, h + (pad_h * 2), w + (pad_w * 2)))
        padded[:, pad_h:-pad_h, pad_w:-pad_w] = images
    for i in range(0, (out_h * out_w)):
        row = i // out_w
        col = i % out_w
        conv[:, row, col] = (
            padded[:, row:kh + row, col:kw + col] * kernel).sum(axis=(1, 2))
    return conv
