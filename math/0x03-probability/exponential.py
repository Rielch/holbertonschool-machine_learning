#!/usr/bin/env python3
"""Exponencial distribution"""


class Exponential:
    """Exponential distribution"""

    e = 2.7182818285

    def __init__(self, data=None, lambtha=1):
        """Initializes an instance of Exponential"""

        self.data = data
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            else:
                self.lambtha = float(lambtha)
        else:
            if type(data) != list:
                raise TypeError("data must be a list")
            elif len(data) <= 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = 1 / (sum(data) / len(data))

    def pdf(self, x):
        """Calculates the PDF for a time period"""

        if x < 0:
            return 0
        return self.lambtha * (self.e ** (-self.lambtha * x))

    def cdf(self, x):
        """Calculates the CDF for a time period"""

        if x < 0:
            return 0
        return 1 - self.e ** (-self.lambtha * x)
