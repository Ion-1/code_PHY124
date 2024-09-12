# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 10:14:23 2024

@author: Ion-1
"""
import scipy
# import numpy as np
import cupy as np
import matplotlib.pyplot as plt
import sympy
from sklearn.utils import Bunch


def polar_to_cartesian(Bunch):
    return ((np.array(Bunch.y[0]) * np.cos(np.array(Bunch.y[1]))).get(), (np.array(Bunch.y[0]) * np.sin(np.array(Bunch.y[1]))).get())


if __name__ == "__main__":
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)
    fig.tight_layout(pad=0)

    r, psi, p_r, p_psi = sympy.symbols("r psi p_r p_psi")
    schwarzschild = (1 - 2 / r) * p_r**2 / 2 + p_psi**2 / 2 / r**2 - r / 2 / (r - 2)
    dr_dt = sympy.lambdify((r, psi, p_r, p_psi), sympy.diff(schwarzschild, p_r), "cupy")
    dpsi_dt = sympy.lambdify((r, psi, p_r, p_psi), sympy.diff(schwarzschild, p_psi), "cupy")
    dp_r_dt = sympy.lambdify((r, psi, p_r, p_psi), -sympy.diff(schwarzschild, r), "cupy")
    dp_psi_dt = sympy.lambdify((r, psi, p_r, p_psi), -sympy.diff(schwarzschild, psi), "cupy")

    def func(t, y):
        return [
            dr_dt(*y),
            dpsi_dt(*y),
            dp_r_dt(*y),
            dp_psi_dt(*y),
        ]

    # Circular orbit
    r_init = 15
    p_psi = r_init**1.5 / (r_init - 2)
    y0 = [r_init, 0, 0, p_psi]
    t_start, t_final = 0, 2400
    result1 = scipy.integrate.solve_ivp(func, (t_start, t_final), y0, max_step=0.5)
    ax1.plot(*polar_to_cartesian(result1))
    ax1.plot(*polar_to_cartesian(Bunch(y=[[15]*100, np.linspace(0, 2*np.pi, 100)])), color='green')

    # Smaller p_psi
    r_init = 15
    p_psi = r_init**1.5 / (r_init - 2)
    y0 = [r_init, 0, 0, p_psi-0.5]
    t_start, t_final = 0, 1500
    result2 = scipy.integrate.solve_ivp(func, (t_start, t_final), y0, max_step=0.5)
    ax2.plot(*polar_to_cartesian(result2))

    # Light rays
    r_init = 10
    p_psi = r_init**1.5 / (r_init - 2)**0.5
    y0 = [r_init, 0, 0, p_psi]
    t_start, t_final = 0, 150
    result3 = scipy.integrate.solve_ivp(func, (t_start, t_final), y0, max_step=0.5)
    ax3.plot(*polar_to_cartesian(result3))
    ax4.plot(result3.t, (np.array(result3.y[1])-np.arccos(r_init/np.array(result3.y[0]))).get())
    ax4.axhline(2/(r_init-2))
    ax3.set(ylim=[-5,80])
    ax4.set(ylim=[0, 2/(r_init-2)+0.05])

    # Initial conditions
    y0 = [55000, 0, 0, 80]
    t_start, t_final = 0, 10**8
    result4 = scipy.integrate.solve_ivp(func, (t_start, t_final), y0, max_step=5000)
    ax5.plot(*polar_to_cartesian(result4))

    # Puzzle
    y0 = [106, 0, 0, 7.88]
    t_start, t_final = 0, 6857
    result5 = scipy.integrate.solve_ivp(func, (t_start, t_final), y0, max_step=0.5)
    print(result5)
    ax6.plot(*polar_to_cartesian(result5))

    # ax4.plot(*polar_to_cartesian(result4))
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    ax3.set_aspect('equal')
    ax5.set_aspect('equal')
    ax6.set_aspect('equal')
