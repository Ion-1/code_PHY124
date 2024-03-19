# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 13:48:37 2023

@author: ion1
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def row_2_list(block):
    return block.rstrip("\n").split("\n")


def plotter(fig, x_vals, y_vals, **kwargs):

    defaultKwargs = {
        "style": "seaborn-v0_8-dark",
        "aspect": "equal",
        "x_label": "",
        "y_label": "",
        "title": "", "title_pad": 10,
        "x_margin": 0.05, "y_margin": 0.05,
        "scatter_style": "bx", "markersize": 8,
        "x_lim": [], "y_lim": [],
        "x_rotation": "horizontal", "x_halignment": "center",
        "y_rotation": "horizontal", "y_halignment": "right",
        "x_scale": "linear", "y_scale": "linear"}

    kwargs = defaultKwargs | kwargs

    fig.clear()

    plt.style.use(kwargs["style"])

    ax = fig.add_subplot()

    x_labels = ax.get_xticklabels()
    y_labels = ax.get_yticklabels()

    plt.setp(x_labels, rotation=kwargs["x_rotation"],
             horizontalalignment=kwargs["x_halignment"])
    plt.setp(y_labels, rotation=kwargs["y_rotation"],
             horizontalalignment=kwargs["y_halignment"])

    x_data_min = min(x_vals)
    x_data_max = max(x_vals)

    y_data_min = min(y_vals)
    y_data_max = max(y_vals)

    x_min_pad = (1-kwargs["x_margin"])*x_data_min
    x_max_pad = (1+kwargs["x_margin"])*x_data_max

    y_min_pad = (1-kwargs["y_margin"])*y_data_min
    y_max_pad = (1+kwargs["y_margin"])*y_data_max

    x_lim = kwargs["x_lim"] if kwargs["x_lim"] else [x_min_pad, x_max_pad]
    y_lim = kwargs["y_lim"] if kwargs["y_lim"] else [y_min_pad, y_max_pad]

    ax = fig.add_subplot()

    ax.set_xlabel(kwargs["x_label"], loc='center',
                  fontstyle='oblique', fontsize='medium')
    ax.set_ylabel(kwargs["y_label"], loc='center',
                  fontstyle='oblique', fontsize='medium')
    ax.set_title(kwargs["title"], loc='center', fontstyle='oblique',
                 fontsize='medium', pad=kwargs["title_pad"])

    ax.set(xlim=x_lim, ylim=y_lim,
           xscale=kwargs["x_scale"], yscale=kwargs['y_scale'])

    ax.plot(x_vals, y_vals, kwargs["scatter_style"],
            markersize=kwargs["markersize"])


if __name__ == "__main__":
    # Code goes here
    test_data_x = [1, 2, 3, 4, 5, 6, 7, 8]
    test_data_y = [1, 1, 2, 3, 5, 8, 13, 21]

    print("The following styles are available:", plt.style.available)
    fig, ax = plt.subplots()
    del ax

    plotter(fig, test_data_x, test_data_y)
