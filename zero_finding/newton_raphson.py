# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 01:16:39 2024

@author: Ion-1
"""

from sympy import symbols, diff, evalf, Float
from sympy import sin, pi
from sympy.utilities.lambdify import lambdify
from math import ceil


def newton_raphson(function, guess, derivative=None, precision=11, max_it = 10000):

    symbol = next(iter(function.free_symbols))

    if derivative is None:
        derivative = diff(function)

    val = Float(guess, precision)
    multi_prec = ceil(precision/2)
    prev_vals = set()

    for _ in range(max_it):
        prev_vals.add(val)
        val -= function.evalf(multi_prec, subs={symbol:val}) / derivative.evalf(multi_prec, subs={symbol:val})
        if val in prev_vals:
            return val

    return val


if __name__ == "__main__":
    x = symbols("x")

    exp = x**3 + x - 62
    diff_exp = diff(exp)

    print(newton_raphson(exp, 1))
