#!/usr/bin/env python3
"""Adds two arrays"""


def add_arrays(arr1, arr2):
    """Adds two arrays and returns a new array with the result"""

    result = []

    if len(arr1) != len(arr2):
        return None

    for element in range(len(arr1)):
        result.append(arr1[element] + arr2[element])

    return result
