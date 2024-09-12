# -*- coding: utf-8 -*-
"""
Created on Wed May 22 07:57:29 2024.

@author: Ion-1
"""
# import numpy as np
import cupy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.artist as art
import matplotlib.transforms as trans
from PIL import Image
from datetime import datetime, date, timedelta, timezone
from functools import partial, update_wrapper
import time

def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func

def daylight(dtime, steps, time_unit="hours", time_step=1) -> np.ndarray:
    """
    Calculate the daylight on an equirectangular projection of earth, shifted based on time.

    Parameters
    ----------
    dtime : datetime.datetime
        Date and time.

    Returns
    -------
    np.ndarray of daylight values between [0, 1].

    """
    x, y = np.meshgrid(
        np.linspace(-180, 180, 360 + 1),
        np.linspace(90, -90, 180 + 1),
        indexing="xy",
    )
    dtime = dtime.astimezone(tz=timezone.utc)

    # init_vector = np.asarray([1, 0, 0])

    # def R_x(t):
    #     return np.asarray([[1, 0, 0], [0, np.cos(t), np.sin(t)], [0, -np.sin(t), np.cos(t)]])

    # def R_y(t):
    #     return np.asarray([[np.cos(t), 0, -np.sin(t)], [0, 1, 0], [np.sin(t), 0, np.cos(t)]])

    # def R_z(t):
    #     return np.asarray([[np.cos(t), np.sin(t), 0], [-np.sin(t), np.cos(t), 0], [0, 0, 1]])

    # def res(x, y, percent_time, angle_year):
    #     return -np.sum(
    #         (init_vector @ R_z(angle_year) @ R_x(-23.5 * np.pi / 180))
    #         * (
    #             init_vector
    #             @ R_y(y * np.pi / 180)
    #             @ R_z(x * np.pi / 180 + 2 * np.pi * percent_time + angle_year)
    #         )
    #     )

    def res(x, y, percent_time, angle_year):
        return (
            np.cos(angle_year)
            * np.cos(y * np.pi / 180)
            * np.cos(x * np.pi / 180 + 2 * np.pi * percent_time + angle_year)
            + np.cos(-23.5 * np.pi / 180)
            * np.sin(angle_year)
            * np.cos(y * np.pi / 180)
            * np.sin(x * np.pi / 180 + 2 * np.pi * percent_time + angle_year)
            - np.sin(y * np.pi / 180) * np.sin(-23.5 * np.pi / 180) * np.sin(angle_year)
        )

    # res = np.vectorize(res, excluded=["percent_time", "angle_year"])
    res = np.vectorize(res)

    for _ in range(steps):
        time = dtime.time()
        percent_time = (
            time.hour + (time.minute + (time.second + time.microsecond / 1000) / 60) / 60
        ) / 24-7/16
        angle_year = (
            2
            * np.pi
            * ((dtime.date() - date(dtime.year, 1, 1)).days)
            / (
                366
                if (dtime.year % 4 == 0 and (dtime.year % 100 != 0 or dtime.year % 400 == 0))
                else 365
            )
        )
        # def ressed(x, y): return (
        #     np.cos(angle_year)
        #     * np.cos(y * np.pi / 180)
        #     * np.cos(x * np.pi / 180 + 2 * np.pi * percent_time + angle_year)
        #     + np.cos(-23.5 * np.pi / 180)
        #     * np.sin(angle_year)
        #     * np.cos(y * np.pi / 180)
        #     * np.sin(x * np.pi / 180 + 2 * np.pi * percent_time + angle_year)
        #     - np.sin(y * np.pi / 180) * np.sin(-23.5 * np.pi / 180) * np.sin(angle_year)
        # )
        # ressed = np.vectorize(ressed)
        # result = ressed(x.ravel(), y.ravel())
        # result = res(x.ravel(), y.ravel(), percent_time, angle_year)
        result = res(x.ravel(), y.ravel(), np.full((65341), percent_time), np.full((65341), angle_year))
        # result = res(x.ravel(), y.ravel(), percent_time, angle_year)
        result[result < 0] = 0
        yield result.reshape((181, 361)), dtime
        dtime = dtime + timedelta(**{f"{time_unit}": time_step})


def _update(i):
    p[0].remove()
    # for c in p[0].collections:
    #     c.remove()
    a, time_ = generator.__next__()
    # p[0] = ax.contourf(a, alpha=0.5)
    p[0] = ax.contourf(a.get(), alpha=0.5)
    timelabel.set_text(f"{time_}")
    t[1:] = t[0:-1]
    t[0] = time.time()
    fpslabel.set_text("{:.3f} fps".format(-1./np.diff(t).mean()))
    return p + [timelabel, fpslabel]


if __name__ == "__main__":
    global p, ax, timelabel, t, fpslabel
    fig, ax = plt.subplots()
    ax.set_aspect("equal")

    img = Image.open("world_map2.png").resize((361, 181))
    d_img = ax.imshow(img, resample=False)
    steps = 10000
    generator = daylight(datetime.now(), steps+1, time_unit="hours", time_step=1)
    p = [0]
    t = np.full(10, time.time())
    a, time_ = generator.__next__()
    # p[0] = ax.contourf(a, alpha=0.5)
    p[0] = ax.contourf(a.get(), alpha=0.5)
    timelabel = ax.text(0.95, 0.9, "", transform=ax.transAxes, ha="right")
    fpslabel = ax.text(0.95, 0.05, "", transform=ax.transAxes, ha="right")
    timelabel.set_text(f"{time_}")
    ani = animation.FuncAnimation(fig, _update, frames=steps, interval=1, blit=True, repeat=False)
