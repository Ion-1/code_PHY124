# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 13:48:37 2023

@author: ion1
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline as spline


def row_2_list(block):
    return block.rstrip("\n").split("\n")


def plotter(fig, ax, x_vals, y_vals, **kwargs):

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

    # ax = fig.add_subplot()
    plt.style.use(kwargs["style"])

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

    ax.grid(True)
    ax.set_xlabel(kwargs["x_label"], loc='center',
                  fontstyle='oblique', fontsize='large')
    ax.set_ylabel(kwargs["y_label"], loc='center',
                  fontstyle='oblique', fontsize='large')
    ax.set_title(kwargs["title"], loc='center', fontstyle='oblique',
                 fontsize='large', pad=kwargs["title_pad"])
    
    # ax.set_xticks()
    # ax.set_yticks()

    ax.set(xlim=x_lim, ylim=y_lim,
           xscale=kwargs["x_scale"], yscale=kwargs['y_scale'])

    ax.plot(x_vals, y_vals, kwargs["scatter_style"],
            markersize=kwargs["markersize"])
    # ax.errorbar()

if __name__ == "__main__":
    flt = np.vectorize(float)
    
    x_ = np.linspace(-90, 95, 1000)

    single_x = list(range(90, -85, -5))
    single_slit = list(flt("""1 0 1 1 2 8 9 6 3 1 4 15 33 38 26 80 220 358 425 410 280 123 28 15 30 25 9 1 1 4 7 6 5 3 1""".split()))
    single_x.reverse(); single_slit.reverse()
    single_spline = spline(single_x, single_slit)

    double_x = list(range(90, -82, -2))
    double_slit = list(flt("""1 1 2 2 0.5 1 3 2 1 3 4 2 2 2 1 1 2 3 7 12 18 24 29 35 37 33 29 19 10 4 4 10 27 52 72 77 78 70 45 18 4 6 23 50 81 96 94 76 50 22 5 1 11 30 47 69 72 62 45 24 8 1 1 5 13 22 28 30 28 24 19 13 8 5 2 1 1 0.5 1 1 1 1 1 1 1 2""".split()))
    double_x.reverse(); double_slit.reverse()
    double_spline = spline(double_x, double_slit)

    # print("The following styles are available:", plt.style.available)
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.tight_layout(pad=4)
    fig.suptitle('Microwave Diffraction    Intensity vs. Angle', size="large")

    kwargs = {"x_lim":[-81,91], "x_label":"Angle [°]", "y_label":"Intensity (indirect) [μA]", "style":"seaborn-v0_8"}

    plotter(fig, ax1, single_x, single_slit, y_lim=[0,440], title="Intensity during single-slit diffraction", scatter_style="bX", markersize=7, **kwargs)
    plotter(fig, ax2, double_x, double_slit, y_lim=[0,100], title="Intensity during double-slit diffraction", scatter_style="bX", markersize=7, **kwargs)
    plotter(fig, ax1, x_, single_spline(x_), y_lim=[0,440], title="Intensity during single-slit diffraction", scatter_style="b-", **kwargs)
    plotter(fig, ax2, x_, double_spline(x_), y_lim=[0,100], title="Intensity during double-slit diffraction", scatter_style="b-", **kwargs)
    
    plt.show()
    plt.savefig('destination_path.eps', format='eps')
