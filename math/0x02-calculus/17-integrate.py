#!/usr/bin/env python3
"""Write a function that calculates the integral of a polynomial:"""


def poly_integral(poly, C=0):
    """ function that calculates the integral of a polynomial:"""

    if type(poly) is not list or not poly or len(poly) == 0:
        return None

    elif type(C) is not int:
        return None

    elif poly == [0]:
        return [C]

    result = []
    result.append(C)
    for x in range(len(poly)):
        result.append(int(poly[x] / (x + 1)))

    return result
