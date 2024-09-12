# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 10:14:02 2024

@author: Ion-1
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import Bunch
from functools import partial
from itertools import combinations


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


def acc_func(y, v, t, M):
    acc = np.zeros((0, 2))
    for planet_index in range(y.shape[0]):
        acc_vec = np.zeros((2))
        for other_index in range(y.shape[0]):
            if planet_index == other_index:
                continue
            vec = y[other_index]-y[planet_index]
            acc_vec += M[other_index] * vec/ np.sum(vec**2)**1.5
        acc = np.append(acc, [acc_vec], axis=0)
    return acc


def pot_calculator(pos, M):
    summation = 0
    for planet_index in range(pos.shape[0]):
        for other_index in range(planet_index):
            distance = np.sum((pos[planet_index]-pos[other_index])**2)**0.5
            summation += M[planet_index]*M[other_index]/distance
    return summation


def kin_calculator(vel, M):
    summation = 0
    for planet_index in range(vel.shape[0]):
        summation += M[planet_index]*np.sum(vel[planet_index]**2)/2
    return summation


def energy_check(Bunch, M):
    diffs = []
    pos = Bunch.positions
    vel = Bunch.velocities

    tot_energy = kin_calculator(vel[0], M) + pot_calculator(pos[0], M)

    for time_index in range(1, pos.shape[0]):
        diff = tot_energy - kin_calculator(vel[time_index], M) - pot_calculator(pos[time_index], M)
        diffs.append(diff)
        if diff > 1e-16:
            print(diff, time_index)



def _main():
    with open("gliese876ini.txt") as file:
        data = np.loadtxt(file).T

    solved = leapfrog(partial(acc_func, M=data[0]), list(zip(data[1],data[2])), list(zip(data[3],data[4])), (0, 3600*24*130, 20*365))

    fig, ax = plt.subplots()

    for planet_index in range(solved.positions.shape[1]):
        ax.plot(solved.positions[:, planet_index, 0], solved.positions[:, planet_index, 1])
    ax.set_aspect('equal')
    energy_check(solved, data[0])
    return solved


if __name__ == "__main__":
    solved = _main()
