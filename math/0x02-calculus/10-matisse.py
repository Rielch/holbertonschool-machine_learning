#!/usr/bin/env python3
"""Write a function def poly_derivative(poly):
that calculates the derivative of a polynomial:"""


def poly_derivative(poly):
    """calculates the derivative of a polynomial"""

    if len(poly) == 0 or type(poly) is not list or not poly:
        return None
    elif len(poly) == 1:
        return [0]

    powers = [i for i in range(len(poly))]
    result = [powers[x] * poly[x] for x in range(1, len(poly))]
    if len(list(result)) == 1:
        return [0]
    return result
