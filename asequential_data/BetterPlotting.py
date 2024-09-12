# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 13:48:37 2023

@author: ion1
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline as spline
from scipy.optimize import curve_fit
from functools import partial


def row_2_list(block):
    return block.rstrip("\n").split("\n")


def plotter(fig, ax, x_vals, y_vals, **kwargs):

    defaultKwargs = {
        "style": "seaborn-v0_8-dark",
        "aspect": "equal",
        "x_label": "",
        "y_label": "",
        "title": "",
        "title_pad": 10,
        "x_margin": 0.05,
        "y_margin": 0.05,
        "scatter_style": "bx",
        "markersize": 8,
        "x_lim": [],
        "y_lim": [],
        "x_rotation": "horizontal",
        "x_halignment": "center",
        "y_rotation": "horizontal",
        "y_halignment": "right",
        "x_scale": "linear",
        "y_scale": "linear",
    }

    kwargs = defaultKwargs | kwargs

    # ax = fig.add_subplot()
    plt.style.use(kwargs["style"])

    x_labels = ax.get_xticklabels()
    y_labels = ax.get_yticklabels()

    plt.setp(x_labels, rotation=kwargs["x_rotation"], horizontalalignment=kwargs["x_halignment"])
    plt.setp(y_labels, rotation=kwargs["y_rotation"], horizontalalignment=kwargs["y_halignment"])

    x_data_min = min(x_vals)
    x_data_max = max(x_vals)

    y_data_min = min(y_vals)
    y_data_max = max(y_vals)

    x_min_pad = (1 - kwargs["x_margin"]) * x_data_min
    x_max_pad = (1 + kwargs["x_margin"]) * x_data_max

    y_min_pad = (1 - kwargs["y_margin"]) * y_data_min
    y_max_pad = (1 + kwargs["y_margin"]) * y_data_max

    x_lim = kwargs["x_lim"] if kwargs["x_lim"] else [x_min_pad, x_max_pad]
    y_lim = kwargs["y_lim"] if kwargs["y_lim"] else [y_min_pad, y_max_pad]

    ax.grid(True)
    ax.set_xlabel(kwargs["x_label"], loc="center", fontstyle="oblique", fontsize="large")
    ax.set_ylabel(kwargs["y_label"], loc="center", fontstyle="oblique", fontsize="large")
    ax.set_title(
        kwargs["title"],
        loc="center",
        fontstyle="oblique",
        fontsize="large",
        pad=kwargs["title_pad"],
    )

    # ax.set_xticks()
    # ax.set_yticks()

    ax.set(xlim=x_lim, ylim=y_lim, xscale=kwargs["x_scale"], yscale=kwargs["y_scale"])
    if "plot_label" in kwargs:
        ax.plot(
            x_vals,
            y_vals,
            kwargs["scatter_style"],
            markersize=kwargs["markersize"],
            label=kwargs["plot_label"],
        )
    else:
        ax.plot(x_vals, y_vals, kwargs["scatter_style"], markersize=kwargs["markersize"])
    # ax.errorbar()


if __name__ == "__main__":

    # x = np.array("""0	30	60	90	120	150	180	210	240	270	300""".split("\t")).astype(int)
    # x = np.concatenate((x, x+330))
    # y = np.array("""321.05	320.35	320.65	320.65	320.45	320.45	320.25	320.15	320.05	320.05	320.5 311.45	311.45	311.45	311.45	311.35	311.35	311.25	311.15	311.25	311.15	311.15""".split()).astype(float)
    # x_top, x_bot = x[:11], x[11:]
    # y_top, y_bot = y[:11], y[11:]
    # def func(x, a, b): return a*x + b
    # top_fit, _ = curve_fit(func, x_top, y_top)
    # bot_fit, _ = curve_fit(func, x_bot, y_bot)
    # top_lin = partial(func, a=top_fit[0], b=top_fit[1])
    # bot_lin = partial(func, a=bot_fit[0], b=bot_fit[1])

    # # print("The following styles are available:", plt.style.available)
    # fig, ax1 = plt.subplots(1, 1)
    # fig.tight_layout(pad=4)
    # # fig.suptitle('Microwave Diffraction    Intensity vs. Angle', size="large")

    # kwargs = {"x_lim":[-1.1,1.1], "x_label":"sin θ", "y_label":"Intensity (indirect) [μA]", "style":"seaborn-v0_8"}

    # plotter(fig, ax1, single_x, single_slit, y_lim=[0,440], title="Intensity during single-slit diffraction", scatter_style="bX", markersize=7, **kwargs)
    # whole_range = np.linspace(kwargs['x_lim'][0], kwargs['x_lim'][1], 1000*(kwargs['x_lim'][1]-kwargs['x_lim'][0]))
    # ax1.grid(True)
    # ax1.set(xlim=kwargs['x_lim'])
    # ax1.set_title("Determining the heat capacity of the container", fontstyle='oblique', fontsize='large', pad=8)
    # ax1.plot(x_top, y_top, 'yo')
    # ax1.plot(x_bot, y_bot, 'ro')
    # ax1.axvline(x = 315, color = 'b', linestyle="--", label="x = 315")
    # ax1.plot(whole_range, top_lin(whole_range), 'y--', label="Before adding cool water")
    # ax1.plot(whole_range, bot_lin(whole_range), 'r--', label="After adding cool water")
    # ax1.scatter([315,315],[top_lin(315),bot_lin(315)], marker='x', s=72, color="black")
    # ax1.set_xlabel("Time [s]")
    # ax1.set_ylabel("Temperature [K]")
    # plt.annotate(f"{top_lin(315):.1f} K", xy=(315+5,top_lin(315)+0.1), horizontalalignment="left", verticalalignment="bottom")
    # plt.annotate(f"{bot_lin(315):.1f} K", xy=(315+5,bot_lin(315)+0.1), horizontalalignment="left", verticalalignment="bottom")
    # plt.legend(loc='lower left')
    # plt.show()
    # plt.savefig(r'C:\Users\ion1\Documents\UZH\FS24\PHY122\VS\LatentFusionGraph.eps', format='eps')
    # x_top = np.array("""0	30	60	90	120	150	180	210	240	270	300""".split("\t")).astype(int)
    # x_bot = np.array("""0	5	10	15	20	25	30	60	90	120	150	180	210	240	270	300""".split("\t")).astype(int)+330
    # x = np.concatenate((x_top, x_bot))
    # y_top = np.array("""322.65	322.65	322.45	322.45	322.25	322.05	322.05	322.05	321.65	321.65	321.55""".split()).astype(float)
    # y_bot = np.array("""317.65	318.25	315.25	315.05	312.35	312.65	311.15	307.35	306.15	305.25	303.85	305.25	304.65	304.45	304.45	305.05""".split()).astype(float)
    # y = np.concatenate((y_top, y_bot))

    # def func(x, a, b): return a*x + b
    # top_fit, _ = curve_fit(func, x_top, y_top)
    # bot_fit, _ = curve_fit(func, x_bot[8:], y_bot[8:])
    # top_lin = partial(func, a=top_fit[0], b=top_fit[1])
    # bot_lin = partial(func, a=bot_fit[0], b=bot_fit[1])
    # tot_lin = spline(np.concatenate(([x[0],x[2],x[4],x[7],x[9], x[13]],x[17:20],x[23:25])), np.concatenate(([y[0],y[2],y[4],y[7],y[9],y[13]],y[17:20],y[23:25])), bc_type="natural")

    # # print("The following styles are available:", plt.style.available)
    # fig, ax1 = plt.subplots(1, 1)
    # fig.tight_layout(pad=4)
    # # fig.suptitle('Microwave Diffraction    Intensity vs. Angle', size="large")

    # kwargs = {"x_lim":[-30,660], "x_label":"Angle [°]", "y_label":"Intensity (indirect) [μA]", "style":"seaborn-v0_8"}

    # # plotter(fig, ax1, single_x, single_slit, y_lim=[0,440], title="Intensity during single-slit diffraction", scatter_style="bX", markersize=7, **kwargs)
    # whole_range = np.linspace(kwargs['x_lim'][0], kwargs['x_lim'][1], 1000*(kwargs['x_lim'][1]-kwargs['x_lim'][0]))
    # ax1.grid(True)
    # ax1.set(xlim=kwargs['x_lim'])
    # ax1.set_title("Determining the latent heat of fusion", fontstyle='oblique', fontsize='large', pad=8)
    # ax1.plot(whole_range, top_lin(whole_range), 'y--', label="Before adding ice")
    # ax1.plot(whole_range, bot_lin(whole_range), 'r--', label="After adding ice")
    # ax1.plot(whole_range, tot_lin(whole_range), 'k-')
    # ax1.plot(x_top, y_top, 'yo')
    # ax1.plot(x_bot, y_bot, 'ro')
    # v = 353
    # ax1.axvline(x = v, color = 'b', linestyle="--", label=f"x = {v}")
    # ax1.scatter([v,v],[top_lin(v),bot_lin(v)], marker='x', s=64, color="black")
    # ax1.set_xlabel("Time [s]")
    # ax1.set_ylabel("Temperature [K]")
    # plt.annotate(f"{top_lin(315):.2f} K", xy=(v+5,top_lin(v)+0.1), horizontalalignment="left", verticalalignment="bottom")
    # plt.annotate(f"{bot_lin(315):.2f} K", xy=(v-5,bot_lin(v)-0.1), horizontalalignment="right", verticalalignment="top")
    # plt.legend(loc='lower left')
    # plt.show()
    # plt.savefig(r'C:\Users\ion1\Documents\UZH\FS24\PHY122\VS\LatentFusionIceGraph.eps', format='eps')
#     fig, (ax1, ax2) = plt.subplots(2, 1)
#     fig.tight_layout(pad=4)
#     fig.suptitle('Microwave Diffraction    Intensity vs. Angle', size="large")

#     single_x = np.flip(
#         np.array(
#             """1
# 0.996194698
# 0.984807753
# 0.965925826
# 0.939692621
# 0.906307787
# 0.866025404
# 0.819152044
# 0.766044443
# 0.707106781
# 0.64278761
# 0.573576436
# 0.5
# 0.422618262
# 0.342020143
# 0.258819045
# 0.173648178
# 0.087155743
# 0
# -0.087155743
# -0.173648178
# -0.258819045
# -0.342020143
# -0.422618262
# -0.5
# -0.573576436
# -0.64278761
# -0.707106781
# -0.766044443
# -0.819152044
# -0.866025404
# -0.906307787
# -0.939692621
# -0.965925826
# -0.984807753""".split()
#         ).astype(float)
#     )
#     single_slit = np.flip(
#         np.array(
#             """1
# 0
# 1
# 1
# 2
# 8
# 9
# 6
# 3
# 1
# 4
# 15
# 33
# 38
# 26
# 80
# 220
# 358
# 425
# 410
# 280
# 123
# 28
# 15
# 30
# 25
# 9
# 1
# 1
# 4
# 7
# 6
# 5
# 3
# 1""".split()
#         ).astype(float)
#     )
#     double_x = np.flip(
#         np.array(
#             """1
# 0.999390827
# 0.99756405
# 0.994521895
# 0.990268069
# 0.984807753
# 0.978147601
# 0.970295726
# 0.961261696
# 0.951056516
# 0.939692621
# 0.927183855
# 0.913545458
# 0.898794046
# 0.882947593
# 0.866025404
# 0.848048096
# 0.829037573
# 0.809016994
# 0.788010754
# 0.766044443
# 0.743144825
# 0.7193398
# 0.69465837
# 0.669130606
# 0.64278761
# 0.615661475
# 0.587785252
# 0.559192903
# 0.529919264
# 0.5
# 0.469471563
# 0.438371147
# 0.406736643
# 0.374606593
# 0.342020143
# 0.309016994
# 0.275637356
# 0.241921896
# 0.207911691
# 0.173648178
# 0.139173101
# 0.104528463
# 0.069756474
# 0.034899497
# 0
# -0.034899497
# -0.069756474
# -0.104528463
# -0.139173101
# -0.173648178
# -0.207911691
# -0.241921896
# -0.275637356
# -0.309016994
# -0.342020143
# -0.374606593
# -0.406736643
# -0.438371147
# -0.469471563
# -0.5
# -0.529919264
# -0.559192903
# -0.587785252
# -0.615661475
# -0.64278761
# -0.669130606
# -0.69465837
# -0.7193398
# -0.743144825
# -0.766044443
# -0.788010754
# -0.809016994
# -0.829037573
# -0.848048096
# -0.866025404
# -0.882947593
# -0.898794046
# -0.913545458
# -0.927183855
# -0.939692621
# -0.951056516
# -0.961261696
# -0.970295726
# -0.978147601
# -0.984807753""".split()
#         ).astype(float)
#     )
#     double_slit = np.flip(
#         np.array(
#             """1
# 1
# 2
# 2
# 0.5
# 1
# 3
# 2
# 1
# 3
# 4
# 2
# 2
# 2
# 1
# 1
# 2
# 3
# 7
# 12
# 18
# 24
# 29
# 35
# 37
# 33
# 29
# 19
# 10
# 4
# 4
# 10
# 27
# 52
# 72
# 77
# 78
# 70
# 45
# 18
# 4
# 6
# 23
# 50
# 81
# 96
# 94
# 76
# 50
# 22
# 5
# 1
# 11
# 30
# 47
# 69
# 72
# 62
# 45
# 24
# 8
# 1
# 1
# 5
# 13
# 22
# 28
# 30
# 28
# 24
# 19
# 13
# 8
# 5
# 2
# 1
# 1
# 0.5
# 1
# 1
# 1
# 1
# 1
# 1
# 1
# 2""".split()
#         ).astype(float)
#     )
#     kwargs = {
#         "x_lim": [-0.985, 1],
#         "x_label": "sin θ",
#         "y_label": "Intensity (indirect) [μA]",
#         "style": "seaborn-v0_8",
#     }
#     spline_range = np.linspace(*kwargs["x_lim"], 10000)
#     single_spline = spline(single_x, single_slit)(spline_range)
#     double_spline = spline(double_x, double_slit)(spline_range)

#     plotter(
#         fig,
#         ax1,
#         single_x,
#         single_slit,
#         y_lim=[0, 440],
#         title="Intensity during single-slit diffraction",
#         scatter_style="bX",
#         markersize=7,
#         plot_label="raw data",
#         **kwargs
#     )
#     ax1.axvline(0, color='red')
#     ax1.axvline(0.423, color='red')
#     ax1.axvline(0.866, color='red')
#     ax1.axvline(-0.500, color='red')
#     ax1.axvline(-0.866, color='red', label='maxima')
#     ax1.axvline(0.342, color='green')
#     ax1.axvline(0.707, color='green')
#     ax1.axvline(-0.423, color='green')
#     ax1.axvline(-0.736, color='green', label='minima')
#     plotter(
#         fig,
#         ax2,
#         double_x,
#         double_slit,
#         y_lim=[0, 100],
#         title="Intensity during double-slit diffraction",
#         scatter_style="bX",
#         markersize=7,
#         plot_label="raw data",
#         **kwargs
#     )
#     ax2.axvline(0, color='red')
#     ax2.axvline(0.309, color='red')
#     ax2.axvline(0.669, color='red')
#     ax2.axvline(-0.375, color='red')
#     ax2.axvline(-0.695, color='red', label='maxima')
#     ax2.axvline(0.174, color='green')
#     ax2.axvline(0.515, color='green')
#     ax2.axvline(0.874, color='green')
#     ax2.axvline(-0.208, color='green')
#     ax2.axvline(-0.545, color='green')
#     ax2.axvline(-0.899, color='green', label='minima')
#     plotter(
#         fig,
#         ax1,
#         spline_range,
#         single_spline,
#         y_lim=[0, 440],
#         title="Intensity during single-slit diffraction",
#         scatter_style="b-",
#         markersize=7,
#         **kwargs
#     )
#     plotter(
#         fig,
#         ax2,
#         spline_range,
#         double_spline,
#         y_lim=[0, 100],
#         title="Intensity during double-slit diffraction",
#         scatter_style="b-",
#         markersize=7,
#         **kwargs
#     )

#     ax1.legend(loc="upper right", framealpha=0.9, frameon=True, facecolor='white')
#     ax2.legend(loc="upper right", framealpha=0.9, frameon=True, facecolor='white')

    fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
    fig2.tight_layout(pad=4)
    fig2.suptitle("Theoretical vs. experimental maxima and minima")

    x = [0, 1, 2]
    single_x_min = [1, 2]

    single_y_max_pos = [0, 0.423, 0.866]
    single_y_max_neg = [0, 0.500, 0.866]
    single_y_max_the = [0, 0.525, 0.875]
    ax1.plot(x, single_y_max_the, 'X-b', label="Theoretical", markersize=8)
    ax1.plot(x, single_y_max_pos, 'X-g', label="Positive", markersize=8)
    ax1.plot(x, single_y_max_neg, 'X-r', label="Negative", markersize=8)
    ax1.set_title("Single-slit maxima experimental vs. theoretical")
    single_y_min_pos = [0.342, 0.707]
    single_y_min_neg = [0.423, 0.736]
    single_y_min_the = [0.350, 0.700]
    ax2.plot(single_x_min, single_y_min_the, 'X-b', label="Theoretical", markersize=8)
    ax2.plot(single_x_min, single_y_min_pos, 'X-g', label="Positive", markersize=8)
    ax2.plot(single_x_min, single_y_min_neg, 'X-r', label="Negative", markersize=8)
    ax2.set_title("Single-slit minima experimental vs. theoretical")
    double_y_max_pos = [0, 0.309, 0.669]
    double_y_max_neg = [0, 0.375, 0.695]
    double_y_max_the = [0, 0.311, 0.622]
    ax3.plot(x, double_y_max_the, 'X-b', label="Theoretical", markersize=8)
    ax3.plot(x, double_y_max_pos, 'X-g', label="Positive", markersize=8)
    ax3.plot(x, double_y_max_neg, 'X-r', label="Negative", markersize=8)
    ax3.set_title("Double-slit maxima experimental vs. theoretical")
    double_y_min_pos = [0.174, 0.515, 0.874]
    double_y_min_neg = [0.208, 0.545, 0.899]
    double_y_min_the = [0.156, 0.467, 0.778]
    ax4.plot(x, double_y_min_the, 'X-b', label="Theoretical", markersize=8)
    ax4.plot(x, double_y_min_pos, 'X-g', label="Positive", markersize=8)
    ax4.plot(x, double_y_min_neg, 'X-r', label="Negative", markersize=8)
    ax4.set_title("Double-slit minima experimental vs. theoretical")

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper left")
    ax3.legend(loc="upper left")
    ax4.legend(loc="upper left")
