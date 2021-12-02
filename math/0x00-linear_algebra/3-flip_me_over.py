#!/usr/bin/env python3
"""Transpose a 2D matrix"""


def matrix_transpose(matrix):
    """Returns the transpose of a 2D matrix"""

    new_matrix = []

    for i in range(len(matrix[0])):
        new_matrix.append([])

    for row in range(len(matrix[0])):
        for element in range(len(matrix)):
            new_matrix[row].append(matrix[element][row])

    return new_matrix
