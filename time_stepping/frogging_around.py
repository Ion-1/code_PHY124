 # -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 09:09:36 2024

@author: Ion-1
"""
import numpy as np
from sympy import symbols, diff, Float, sympify
from decimal import Decimal
from sklearn.utils import Bunch


def euler_1D(diffeq, start_vals: tuple, t_range: tuple, t_sym_name: str = "t", step: float = 0.01):
    x, v_x = start_vals
    x_symbol, t_symbol = None, None
    for sym in iter(diffeq.free_symbols):
        if str(sym) == t_sym_name:
            t_symbol = sym
        else:
            x_symbol = sym

    for t in np.linspace(t_range[0], t_range[1], int((t_range[1] - t_range[0]) / step)):
        x_old, v_x_old = x, v_x
        x += step * v_x_old
        v_x += step * diffeq.evalf(10, subs={t_symbol: t, x_symbol: x_old})

    return x, v_x


def leapfrog_multD_linear(
    diffeq_hom,
    diffeq_inhom,
    start_vals: tuple,
    t_range: tuple,
    x_names: list = [""],
    t_sym_name: str = "t",
    step: float = 0.01,
):
    x, v_x = start_vals

    for t in np.linspace(t_range[0], t_range[1], int((t_range[1] - t_range[0]) / step)):
        x_old, v_x_old = x, v_x
        x += step * v_x_old + step**2 * (
            diffeq_hom.subs(t_sym_name, t) @ x_old + diffeq_inhom.subs(t_sym_name, t)) / 2
        v_x += (
            (
                diffeq_inhom.subs(t_sym_name, t) + diffeq_hom.subs(t_sym_name, t) @ x_old
                + diffeq_inhom.subs(t_sym_name, t + step) + diffeq_hom.subs(t_sym_name, t+step) @ x
            )
            * step
            / 2
        )

    return x, v_x

def leapfrog(f, x0, v0, t):
    """
    Integrate using the leapfrog algorithm with (optional) variable time-steps.

    Parameters
    ----------
    f : function
        Right side of the diff equation. Called with current position and time as arrays.
        Called as: f(y, v, t), where y is position, v is velocity and t is time
    x0 : array-like
        Initiial values for position.
    v0 : array-like
        Initial values for velocity
    t : array-like
        Takes (t_start, t_final, num_steps) or the time-values to be integrated over (len > 3).
        num_steps is optional, default is (t_final-t_start)*100

    Returns
    -------
    sklearn.utils.Bunch
        Bunch of times, positions and velocities.

    """

    # TODO: Move away from float

    pos = np.array([x0])
    vel = np.array([v0])
    time = t[0]
    times = np.array([time])
    acc = f(pos[-1], vel[-1], time)

    time_steps = (
        t
        if len(t) > 3
        else [(t[1] - t[0]) / t[2] if len(t) == 3 else (t[1] - t[0]) / 100]
        * (t[2] if len(t) == 3 else 100)
    )

    for step in time_steps:
        time += step
        times = np.append(times, time)
        mid_vel = vel[-1] + acc * step / 2
        pos = np.append(pos, [pos[-1] + mid_vel * step], axis=0)
        acc = f(pos[-1], vel[-1], time)
        vel = np.append(vel, [mid_vel + acc * step / 2], axis=0)

    return Bunch(positions=pos, velocities=vel, times=times)

if __name__ == "__main__":

    a_x = sympify(0)
    a_y = sympify(-9.81)
    t_final = 2.7

    x_t = euler_1D(a_x, (0, 9), (0, t_final))[0]
    y_t = euler_1D(a_y, (0, 11), (0, t_final))[0]
    print(x_t, y_t)

    a_mat = sympify(np.array([a_x, a_y]))
    print(leapfrog_multD_linear(sympify(np.array([[0,0],[0,0]])), a_mat, (np.array([0, 0], dtype=Float), np.array([9, 11], dtype=Float)), (0, t_final)))
