#!/usr/bin/env python3
"""Write a function def summation_i_squared(n):
 that calculates sum_{i=1}^{n} i^2"""


def summation_i_squared(n):
    """Calculates sum_{i=1}^{n}i^2"""

    if type(n) is not int or n < 1:
        return None

    return int(n * (n + 1) * (2 * n + 1) / 6)
