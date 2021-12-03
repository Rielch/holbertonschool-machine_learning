#!/usr/bin/env python3
"""Performs matrix multiplication"""


def mat_mul(mat1, mat2):
    """Multiplicates two matrices and returns a new matrix with the result"""

    result = []

    if len(mat1[0]) != len(mat2):
        return None

    for row in range(len(mat1)):
        result.append([])
        for element in range(len(mat2[0])):
            result[row].append(0)

    for a in range(len(mat1)):
        for b in range(len(mat2[0])):
            for c in range(len(mat2)):
                result[a][b] += mat1[a][c] * mat2[c][b]

    return result
