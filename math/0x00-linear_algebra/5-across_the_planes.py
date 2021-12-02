#!/usr/bin/env python3
"""Adds two arrays"""


def add_matrices2D(arr1, arr2):
    """Adds two arrays and returns a new array with the result"""

    result = []

    if len(arr1) != len(arr2) or len(arr1[0]) != len(arr2[0]):
        return None

    for row in range(len(arr1)):
        result.append([])
        for element in range(len(arr1[0])): 
            result[row].append(arr1[row][element] + arr2[row][element])

    return result
