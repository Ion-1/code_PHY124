# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 08:55:56 2024

@author: Ion-1
"""
from fractions import Fraction


def compute_pi(n: int) -> str:
    """
    Calculates pi (circumference/diameter of circle) using the taylor expansion
    of arctan.

    Parameters
    ----------
    n : int
        Number of decimal places pi should be calculated to.

    Returns
    -------
    str
        (n+1) digits of pi.

    """

    def arctan(x: Fraction, n: int):

        counter = 0
        term = x
        result = x

        while abs(term*10**n) > 1:

            counter += 1
            term *= -x**2
            result += term*Fraction(1, 2*counter+1)

        return result

    if n > 4298:
        return bin(int(4*(arctan(Fraction(1, 2), n)+arctan(Fraction(1, 3), n))*10**n))
    return format(4*(arctan(Fraction(1, 2), n)+arctan(Fraction(1, 3), n)), f".{n}f")
