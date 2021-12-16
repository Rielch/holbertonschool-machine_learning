#!/usr/bin/env python3
"""Poisson distribution"""


class Poisson:
    """Poison distribution"""

    e = 2.7182818285

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

        def factorial(n):
            if n == 1 or n == 0:
                return 1
            else:
                return n * factorial(n-1)

        k = int(k)
        if k < 0:
            return 0
        return (self.lambtha ** k * self.e ** -self.lambtha) / factorial(k)

    def cdf(self, k):
        """Calculates the CDF for a given number of successes"""

        if not isinstance(k, int):
            k = int(k)
        if k <= 0:
            return 0
        result = 0
        for i in range(0, k + 1):
            result += self.pmf(i)
        return result
