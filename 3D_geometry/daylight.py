# -*- coding: utf-8 -*-
"""
Created on Wed May 22 07:57:29 2024.

@author: Ion-1
"""
import numpy as np
# import cupy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
from datetime import datetime, date, timedelta, timezone
import time

from typing import Generator

def daylight(dtime: datetime, steps: int, time_unit: str = "hours", time_step: int = 1, module: {"numpy", "cupy"} = "cupy") -> Generator[np.ndarray, None, None]:
    """
    Calculate the daylight on an equirectangular projection of earth, shifted based on time.

    Parameters
    ----------
    dtime : datetime.datetime
        Date and time.
    steps : int
        How many frames will be rendered.
    time_unit : str
        Unit of the time step between frames.
    time_step : int
        How many time_unit between frames.
    module : {"numpy", "cupy"}
        Whether to use numpy or cupy.

    Returns
    -------
    Generator of np.ndarray with daylight values between [0, 1].

    """
    if module == "numpy":
        import numpy as np
    elif module == "cupy":
        import cupy as np
    else:
        raise ValueError("Bad value for module")

    x, y = np.meshgrid(
        np.linspace(-np.pi, np.pi, 360 + 1),
        np.linspace(np.pi/2, -np.pi/2, 180 + 1),
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
            * np.cos(y)
            * np.cos(x + 2 * np.pi * percent_time + angle_year)
            + np.cos(-23.5 * np.pi / 180)
            * np.sin(angle_year)
            * np.cos(y)
            * np.sin(x + 2 * np.pi * percent_time + angle_year)
            - np.sin(y) * np.sin(-23.5 * np.pi / 180) * np.sin(angle_year)
        )

    if module == "numpy":
        res = np.vectorize(res, excluded=["percent_time", "angle_year"])
    else:
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
        if module == "numpy":
            result = res(x.ravel(), y.ravel(), percent_time, angle_year)
            result[result < 0] = 0
            yield result.reshape((181, 361)), dtime
        else:
            result = res(x.ravel(), y.ravel(), np.full((65341), percent_time), np.full((65341), angle_year))
            result[result < 0] = 0
            yield result.reshape((181, 361)).get(), dtime
        dtime = dtime + timedelta(**{f"{time_unit}": time_step})


def _update(i):
    p[0].remove()
    # for c in p[0].collections:
    #     c.remove()
    a, time_ = generator.__next__()
    # p[0] = ax.contourf(a, alpha=0.5)
    p[0] = ax.contourf(a, alpha=0.5)
    timelabel.set_text(f"{time_}")
    t[1:] = t[0:-1]
    t[0] = time.time()
    fpslabel.set_text("{:.3f} fps".format(-1./np.diff(list(filter(lambda x: x is not None, t))).mean()))
    return p + [timelabel, fpslabel]


if __name__ == "__main__":
    global p, ax, timelabel, t, fpslabel
    fig, ax = plt.subplots()
    ax.set_aspect("equal")

    img = Image.open("world_map2.png").resize((361, 181))
    d_img = ax.imshow(img, resample=False)
    frames = 1000
    generator = daylight(datetime.now(), frames + 1, time_unit="hours", time_step=1)
    p = [0]
    t = np.full(100, None)
    t[0] = time.time()
    a, time_ = generator.__next__()
    # p[0] = ax.contourf(a, alpha=0.5)
    p[0] = ax.contourf(a, alpha=0.5)
    timelabel = ax.text(0.95, 0.9, "", transform=ax.transAxes, ha="right")
    fpslabel = ax.text(0.95, 0.05, "", transform=ax.transAxes, ha="right")
    timelabel.set_text(f"{time_}")
    ani = animation.FuncAnimation(fig, _update, frames=frames, interval=1, blit=True, repeat=False)
    plt.show()
