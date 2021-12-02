#!/usr/bin/env python3
"""Calculates the shape of a matrix"""


def matrix_shape(matrix):
    """Calculates the shape of a given matrix
    Returns a list with the values"""
    shape = []

    while type(matrix) == list:
        shape.append(len(matrix))
        matrix = matrix[0]

    return shape
