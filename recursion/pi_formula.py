# -*- coding: utf-8 -*-
"""
Created on Thu May 23 17:33:26 2024.

@author: Ion-1
"""
import numpy as np
import cupy as cp
from fractions import Fraction


def arctan(x: Fraction, steps: int) -> Fraction:

    counter = 0
    term = x
    result = x

    for _ in [0]*steps:

        counter += 1
        term *= -x**2
        result += term*Fraction(1, 2*counter+1)

    return result


def tann(n: int, tan_alpha: Fraction) -> Fraction:
    if n == 1:
        return tan_alpha
    nex = tann(n-1, tan_alpha)
    return (nex+tan_alpha)/(1-nex*tan_alpha)


if __name__ == "__main__":
    digits_precision = 15
    a = Fraction(1, 5)
    N = 2
    tan_n = tann(N, a)
    b = (1 - tan_n)/(1 + tan_n)
    pi = 4 * (N * arctan(a, int(digits_precision/np.log10(float(1/a)))+1) + arctan(b, int(digits_precision/np.log10(float(1/a)))+1))
    print(a, b, f"{pi:.10f}")
