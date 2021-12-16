#!/usr/bin/env python3
"""Binomial distribution"""


class Binomial:
    """Binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        """Initializes a Binomial instance"""

        self.data = data
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            else:
                self.n = int(n)
                self.p = float(p)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) <= 2:
                raise ValueError("data must contain multiple values")
            mean = float(sum(data) / len(data))
            ns = 0
            for i in data:
                ns = ns + ((i - mean) ** 2)
            self.n = round(mean ** 2 / (mean - (ns / len(data))))
            self.p = float(mean / self.n)

    def pmf(self, k):
        """Calculates the PMF for a number of successes"""

        def factorial(n):
            """Returns the factorial of a number"""
            result = 1
            for i in range(2, k + 1):
                result = result * i
            return result

        k = int(k)
        if k < 0:
            return 0
        return ((factorial(self.n) /
                 (factorial(k) * factorial(self.n - k))) *
                (self.p ** k) * ((1 - self.p) ** (self.n - k)))

    def cdf(self, k):
        """Calculates the CDF for a number of successes"""

        k = int(k)
        if k < 0:
            return 0
        result = 0
        for i in range(0, k + 1):
            result += self.pmf(i)
        return result
