#!/usr/bin/env python3
"""Concatenates two matrices along a especific axis"""


def cat_matrices2D(mat1, mat2, axis=0):
    """Creates a new matrix and concatenates the two given matrices in it"""

    new_matrix = []

    if axis == 0 and len(mat1[0]) == len(mat2[0]):
        new_matrix = list(mat1)
        for element in mat2:
            new_matrix.append(element)

    elif axis == 1 and len(mat1) == len(mat2):
        for element in range(len(mat2)):
            new_matrix.append(mat1[element] + mat2[element])

    else:
        return None

    return new_matrix
