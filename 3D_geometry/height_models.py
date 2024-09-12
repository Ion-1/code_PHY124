# -*- coding: utf-8 -*-
"""
Created on Wed May  8 07:57:18 2024.

@author: Ion-1
"""
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    grid_shape = (int(1000 / 0.5), int(1000 / 0.5))

    terrain_file = "data/terrain.xyz"
    surface_file = "data/surface.xyz"

    t_array = np.loadtxt(terrain_file, dtype=float, skiprows=1).T
    t_arrays = t_array.reshape(3,2000,2000)

    s_array = np.loadtxt(surface_file, dtype=float, skiprows=1).T
    s_arrays = s_array.reshape(3,2000,2000)

    fig = plt.figure()

    t_ax = fig.add_subplot(1, 3, 1)
    s_ax = fig.add_subplot(1, 3, 3)
    d_ax = fig.add_subplot(1, 3, 2)

    t_ax.set_aspect(1)
    s_ax.set_aspect(1)
    d_ax.set_aspect(1)

    t_ax.set_title("Terrain contour")
    s_ax.set_title("Surface contour")
    d_ax.set_title("Difference contour")

    t_ax.contourf(*t_arrays)
    s_ax.contourf(*s_arrays)

    assert np.all(t_arrays[0] == s_arrays[0]) and np.all(t_arrays[1] == s_arrays[1])

    d_ax.contourf(t_arrays[0], t_arrays[1], s_arrays[2] - t_arrays[2])

    plt.subplots_adjust(left=0.04, right=0.98)
    plt.show()
