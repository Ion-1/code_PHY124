# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 16:05:23 2024.

@author: Ion-1
"""

import numpy as np
import os
import csv
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import OrderedDict


def dictify(
    reader,
    header_index=0,
    zero_value="0",
    filter_=lambda index, row: True,
    **kwargs,
):

    indices = {}
    data = OrderedDict() if "OrderedDict" in dir() else {}

    for index, row in enumerate(reader):

        if index < header_index:
            continue

        if index == header_index:
            for key, item in kwargs.items():
                indices[key] = row.index(item)
            continue

        if not filter_(index, row):
            continue

        data[row[indices["key"]]] = [
            (row[indices[f"val{i}"]] if row[indices[f"val{i}"]] else zero_value)
            for i in range(len(kwargs) - 1)
        ]

    return data


filename = "V-Dem-CY-Core-v14_csv\\V-Dem-CY-Core-v14.csv"

current_wdir = os.getcwd()

path = filename if os.path.isabs(filename) else os.path.join(current_wdir, filename)

if not os.path.exists(path):
    raise FileNotFoundError(f"path {path} does not exist")

with open(path, "r", newline="") as csvfile:

    reader = csv.reader(csvfile)

    data = dictify(
        reader,
        0,
        zero_value=None,
        key="year",
        filter_=lambda index, row: row[0] == "Taiwan",
        val0="v2x_libdem",
        val1="v2exbribe",
        val2="v2exembez",
        val3="v2x_corr",
    )

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, squeeze=True)

fig.suptitle("Democracy and corruption in Taiwan")

x_data, y_data, years = zip(
    *[(float(data[key][0]), float(data[key][3]), int(key)) for key in data]
)
ratio = [y / x for x, y in zip(x_data, y_data)]

ax1.set(
    xlabel="Democracy index",
    ylabel="Corruption index",
    xscale="linear",
    yscale="linear",
    xlim=(0, 1),
    ylim=(0, 1),
)
# ax1.set_xticks([0.1, 0.5, 0.9])
ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
# ax1.set_yticks([0.1, 0.5, 0.9])
ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

ax2.set(
    xlabel="Year",
    ylabel="Democracy vs. Corruption",
    xlim=(1900, 2025),
    ylim=(0.95 * min(ratio), 1.05 * max(ratio)),
)

scat = ax1.plot(x_data[0], y_data[0], "bo", markersize=7, label="Scattered democracy")[0]
scat2 = ax1.plot(x_data[0], y_data[0], "go", markersize=7)[0]
prev_ann = ax1.annotate(
    f"{years[0]}",
    (x_data[0], y_data[0]),
    fontsize=13,
    fontstyle="italic",
    weight="bold",
    color="green",
    # backgroundcolor="white",
    textcoords="offset points",
    xytext=(23, 4),
    ha="center",
)

line = ax2.plot(years[0], ratio[0], "go-", markersize=5, label="Chronological view")[0]


def update(frame):

    if frame != 0:
        global prev_ann
        prev_ann.remove()

        line.set_xdata(years[: frame + 1])
        line.set_ydata(ratio[: frame + 1])

        scat.set_xdata(x_data[: frame + 1])
        scat.set_ydata(y_data[: frame + 1])
        scat2.set_xdata([x_data[frame]])
        scat2.set_ydata([y_data[frame]])

        prev_ann = ax1.annotate(
            f"{years[frame]}",
            (x_data[frame], y_data[frame]),
            fontsize=13,
            fontstyle="italic",
            weight="bold",
            color="green",
            # backgroundcolor="white",
            textcoords="offset points",
            xytext=(23 if x_data[frame] < 0.8 else -28, 4 if y_data[frame] < 0.8 else -8),
            ha="center",
        )

    return (scat, line)


ani = animation.FuncAnimation(
    fig=fig,
    func=update,
    frames=len(years),
    interval=500,
    repeat_delay=3000,
)

plt.show()