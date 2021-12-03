#!/usr/bin/env python3
"""Concatenates two arrays"""


def cat_arrays(arr1, arr2):
    """Creates a new array and concatenates the two arrays in it"""

    new_array = list(arr1)

    for element in arr2:
        new_array.append(element)

    return new_array
