#!/usr/bin/env python3
"""Write a function def poly_derivative(poly):
that calculates the derivative of a polynomial:"""


def poly_derivative(poly):
    """calculates the derivative of a polynomial"""

    if len(poly) == 0:
        return None
    elif len(poly) == 1:
        return [0]

    result = []
    for x in range(1, len(poly)):
        result.append(poly[x] * x)

    return result
