#!/usr/bin/env python3
"""Poisson distribution"""


class Poisson:
    """Poison distribution"""

    def __init__(self, data=None, lambtha=1):
        """Initializates an instance of Poisson"""

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
            self.lambtha = float(sum(data) / len(data))


    def pmf(self, k):
        """Calculates the PMF for a given number of successes"""

        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        factorial = 1
        if k >= 1:
            for i in range(1, k + 1):
                factorial = factorial * i
        return (2.7182818285 ** (- self.lambtha)) * (self.lambdtha ** k) / factorial

    def cdf(self, k):
        """Calculates the CDF for a given number of successes"""

        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        result = 0
        for i in range(0, k + 1):
            result += self.pmf(i)
        return cdf
