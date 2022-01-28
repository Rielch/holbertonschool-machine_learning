#!/usr/bin/env python3
"""function that performs a same convolution on grayscale images"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """Performs a same convolution on grayscale images"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    conv = np.zeros(shape=(m, h, w))
    if kh % 2:
        pad_h = (kh - 1) // 2
    else:
        pad_h = kh // 2
    if kw % 2:
        pad_w = (kw - 1) // 2
    else:
        pad_w = kw // 2
    padded = np.zeros(shape=(m, h + (pad_h * 2), w + (pad_w * 2)))
    padded[:, pad_h:-pad_h, pad_w:-pad_w] = images
    for i in range(0, (h * w)):
        row = i // w
        col = i % w
        conv[:, row, col] = (
            padded[:, row:kh + row, col:kw + col] * kernel).sum(axis=(1, 2))
    return conv
