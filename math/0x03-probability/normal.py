#!/usr/bin/env python3
"""Normal distribution"""


class Normal:
    """Normal distribution"""

    e = 2.7182818285
    pi = 3.1415926536
    tau = 2 * pi

    def __init__(self, data=None, mean=0., stddev=1.):
        """Initialize an instance of Normal"""

        self.data = data
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            else:
                self.mean = float(mean)
                self.stddev = float(stddev)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) <= 2:
                raise ValueError("data must contain multiple values")
            self.mean = float(sum(data) / len(data))
            difference = 0
            for i in range(len(data)):
                difference += (data[i] - self.mean) ** 2
            self.stddev = (difference / len(data)) ** 0.5

    def erf(self, x):
        """Error function"""

        return (2 / (self.pi ** 0.5)) * (x
                                         - x ** 3 / 3
                                         + x ** 5 / 10
                                         - x ** 7 / 42
                                         + x ** 9 / 216)

    def z_score(self, x):
        """Calculates the z-score of a x-value"""

        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calcualtes the x-value of a z-score"""

        return (z * self.stddev) + self.mean

    def pdf(self, x):
        """Calculates the PDF of a x-value"""

        return 1 / (self.stddev * self.tau ** 0.5 * self.e **
                    (self.z_score(x) ** 2 / 2))

    def cdf(self, x):
        """Calculates the CDF of a x-value"""

        return (1 + self.erf((x - self.mean) / (self.stddev * 2 ** 0.5))) / 2
