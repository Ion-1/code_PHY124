# -*- coding: utf-8 -*-
"""
Created on Thu May 23 08:18:43 2024.

@author: Ion-1
"""
import numpy as cp
# import cupy as cp
import cupyx
import matplotlib.pyplot as plt
import time
from collections import OrderedDict


def x_coord(n):
    return cp.round(cp.sum(cp.sin(cp.pi*cp.floor(cp.sqrt(4*cp.arange(0, n, 1)+1))/2)))


def y_coord(n):
    return cp.round(cp.sum(cp.cos(cp.pi*cp.floor(cp.sqrt(4*cp.arange(0, n, 1)+1))/2)))


def inverse_parameterization(shape, x_coord, y_coord):
    parameters = cp.arange(0, shape[0]*shape[1])
    dic = OrderedDict()
    for n in parameters:
        dic[(int(x_coord(n)), int(y_coord(n)))] = n
    return dic


def xkcd_grid(max_size=15, start_par=4, end_par=1):
    resistances = []
    core = -1
    max_params = (2*max_size+1)*(2*max_size)
    big_dict = inverse_parameterization((2*max_size+3, 2*max_size+2), x_coord, y_coord)
    core_array = cp.zeros((max_params, max_params))
    for size in cp.arange(1, max_size+1):
        size = int(size)
        num_params = (2*size+1)*(2*size)
        big_array = core_array[:num_params, :num_params]
        for coord in list(big_dict.keys())[int(core+1):int(num_params)]:
            param = big_dict[(int(coord[0]), int(coord[1]))]
            counter = 0
            edges = [big_dict[(int(coord[0]) - 1, int(coord[1]))],
                     big_dict[(int(coord[0]) + 1, int(coord[1]))],
                     big_dict[(int(coord[0]), int(coord[1]) - 1)],
                     big_dict[(int(coord[0]), int(coord[1]) + 1)]]

            for edge in filter(lambda par: par < num_params, edges):
                big_array[param, edge] = -1
                counter += 1

            big_array[param, param] = counter
        current_array = cp.zeros((num_params))
        current_array[start_par] = 1
        current_array[end_par] = -1
        voltages = cp.linalg.solve(big_array, current_array)
        resistances.append(float(voltages[start_par]-float(voltages[end_par])))
    return resistances


if __name__ == "__main__":
    fig, ax = plt.subplots()
    max_size = 30
    start = time.perf_counter()
    result = xkcd_grid(max_size)
    print(f"All resistances up to matrix size {2*max_size+1}x{2*max_size} took {time.perf_counter()-start:.2f}s")
    # print(timeit.timeit(lambda: main(max_size), number=3))
    # print(result)
    # resistances = [res[0] for res in result]
    ax.plot(list(range(1, max_size+1)), result)
