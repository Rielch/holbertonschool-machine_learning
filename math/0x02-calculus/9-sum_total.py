#!/usr/bin/env python3
"""Write a function def summation_i_squared(n): that calculates \sum_{i=1}^{n} i^2"""


def summation_i_squared(n):
    """Calculates \sum_{i=1}^{n}i^2"""

    if type(n) is not int or n < 1:
        return None

    if n == 1:
        return 1

    return summation_i_squared(n - 1) + n**2
