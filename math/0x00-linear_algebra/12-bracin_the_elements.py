#!/usr/bin/env python3
"""Performs element-wise addition, substraction, multiplication and division"""
import numpy as np


def np_elementwise(mat1, mat2):
    """Performs addition, substraction, multiplication and division"""
    return (np.add(mat1, mat2), np.subtract(mat1, mat2),
            np.multiply(mat1, mat2), np.divide(mat1, mat2))
