# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 10:34:10 2024

@author: Ion-1
"""

from sympy import symbols, diff, nsolve, Float
from sympy import sin, cos, sinh, cosh, pi
from sympy.utilities.lambdify import lambdify

# from sympy.solvers import solve
import matplotlib.pyplot as plt
import numpy as np
from newton_raphson import newton_raphson as newson


def plt_halleys_comet(fig, ax, rnge=(0, 2 * np.pi), dots=200):
    t, p = symbols("t p")
    e = 0.967
    a = 17.8
    eq = p - e * sin(p)
    x = lambdify(p, a * (cos(p) - e), "numpy")
    y = lambdify(p, a * (1 - e**2) ** 0.5 * sin(p), "numpy")
    t_range = np.linspace(rnge[0], rnge[1], dots)
    phi_range = [
        float(newson(eq - t, 0)) for t in t_range # TODO Gotta fix this shit. Sympy floats don't work with sympy trig functions. Makes sense.
    ]
    ax.plot(x(phi_range), y(phi_range), "yo-")


def plt_oumuamua(fig, ax, rnge=(-np.pi, np.pi), dots=70):
    t, p = symbols("t p")
    e = 1.20
    a = 1.28
    eq = e * sinh(p) - p
    x = lambdify(p, a * (e - cosh(p)), "numpy")
    y = lambdify(p, a * (e**2 - 1) ** 0.5 * sinh(p), "numpy")
    t_range = np.linspace(rnge[0], rnge[1], dots)
    phi_range = [
        float(newson(eq - t, 0)) for t in t_range
    ]
    ax.plot(x(phi_range), y(phi_range), "yo-")


if __name__ == "__main__":
    plt.style.use("dark_background")

    fig, (ax1, ax2) = plt.subplots(1, 2)

    plt_halleys_comet(fig, ax1)
    plt_oumuamua(fig, ax2)

    plt.show()
