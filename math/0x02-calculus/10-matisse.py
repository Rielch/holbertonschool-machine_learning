#!/usr/bin/env python3
"""Write a function def poly_derivative(poly):
that calculates the derivative of a polynomial:"""


def poly_derivative(poly):
    """calculates the derivative of a polynomial"""

    if type(poly) is not list or not poly or len(poly) == 0:
        return None
    elif len(poly) == 1:
        return [0]

    result = []
    for x in range(1, len(poly)):
        result.append(x * poly[x])
    if len(list(result)) == 1:
        return [0]
    return result
