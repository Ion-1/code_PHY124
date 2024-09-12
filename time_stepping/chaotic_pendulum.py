# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 08:49:20 2024.
https://www.twitch.tv/wirtual/clip/WittyProudNeanderthalNerfBlueBlaster-jJYa1FZTaHTpnU8v
@author: Ion-1
"""

import os
import sys
import logging
import multiprocessing
import numpy as np
import vispy
import functools
import time
import pickle
import imageio
import imageio.v3 as iio
from pygifsicle import optimize
import matplotlib.pyplot as plt
from PIL import Image
from vispy import app, scene

# Dev modules, delete eventually
import timeit

# vispy.use("pyglet")


class Gradient:
    """Makes a color gradient, based on https://stackoverflow.com/a/49321304."""

    def __init__(self, color1, color2, gamma=0.43):
        self.gamma = gamma
        self.color1_lin = tuple(self._from_sRGB(c) for c in color1)
        self.bright1 = sum(self.color1_lin) ** gamma
        self.color2_lin = tuple(self._from_sRGB(c) for c in color2)
        self.bright2 = sum(self.color2_lin) ** gamma

    @staticmethod
    def _from_sRGB(x):
        x /= 255.0
        if x <= 0.04045:
            y = x / 12.92
        else:
            y = ((x + 0.055) / 1.055) ** 2.4
        return y

    @staticmethod
    def _to_sRGB(c):
        return int(
            255.9999 * (12.92 * c if c <= 0.0031308 else (1.055 * (c ** (1 / 2.4))) - 0.055)
        )

    def color(self, ratio):
        """Return a sRGB color for the ratio between color 1 and 2."""
        intensity = (self.bright1 * (1 - ratio) + self.bright2 * ratio) ** (1 / self.gamma)
        color = tuple(
            (c1 * (1 - ratio) + c2 * ratio) ** (1 / self.gamma)
            for c1, c2 in zip(self.color1_lin, self.color2_lin)
        )
        if sum(color) != 0:
            color = [c * intensity / sum(color) for c in color]
        return [self._to_sRGB(c) for c in color]


class Drawer:
    """
    Draws lines. Pretty much.

    Great resource: http://kt8216.unixcab.org/murphy/index.html.
    """

    def __init__(self, wheel_center, wheel_radius, pendulum_length, dots_per_cm):
        self.wheel_center = wheel_center
        self.wheel_radius = wheel_radius
        self.pendulum_length = pendulum_length
        self.dots_per_cm = dots_per_cm

    def circle(self, thickness=3):
        """Draw a circle."""

        def draw_circle(xc, yc, x, y):
            return np.array(
                [
                    [xc + x, yc + y],
                    [xc - x, yc + y],
                    [xc + x, yc - y],
                    [xc - x, yc - y],
                    [xc + y, yc + x],
                    [xc - y, yc + x],
                    [xc + y, yc - x],
                    [xc - y, yc - x],
                ]
            )

        array = np.zeros((0, 2), dtype=int)
        filled_array = np.zeros((0, 2), dtype=int)
        y = 0
        while y <= self.wheel_radius * self.dots_per_cm + thickness:
            x = 0
            while x <= y:
                rad = (x**2 + y**2) ** 0.5
                if rad <= self.dots_per_cm * self.wheel_radius + thickness + 0.5:
                    filled_array = np.concatenate(
                        (
                            filled_array,
                            draw_circle(self.wheel_center[0], self.wheel_center[1], x, y), 
                        ), axis = 0
                    )
                    if self.dots_per_cm * self.wheel_radius - thickness - 0.5 <= rad:
                        array = np.concatenate(
                            (array, draw_circle(self.wheel_center[0], self.wheel_center[1], x, y)), axis = 0
                        )
                x += 1
            y += 1

        return array, filled_array

    def line(self, angle, tau, thickness=0):
        """Draw a line."""
        init_x = self.wheel_center[0] + int(self.wheel_radius * self.dots_per_cm * np.sin(tau))
        end_x = init_x + int(self.pendulum_length * self.dots_per_cm * np.sin(angle))
        init_y = self.wheel_center[1] - int(self.wheel_radius * self.dots_per_cm * np.cos(tau))
        end_y = init_y - int(self.pendulum_length * self.dots_per_cm * np.cos(angle))

        if thickness == 0:
            return Drawer.bresenline(init_x, init_y, end_x, end_y)
        return Drawer.thick_bresenline(init_x, init_y, end_x, end_y, thickness)

    @staticmethod
    def _bresenline_wrapper(func):
        @functools.wraps(func)
        def wrapper(*args):
            # Input: x0, y0, x1, y1, thick (optional)
            # x0, x1, y0, dx, dy, thick
            dx = args[2] - args[0]
            dy = args[3] - args[1]
            xstep, dx = (1, dx) if dx > 0 else (-1, -dx)
            ystep, dy = (1, dy) if dy > 0 else (-1, -dy)

            if abs(dy) < abs(dx):
                flip = False
                new_args = [args[0], args[1], dx, dy, xstep, ystep]
            else:
                flip = True
                new_args = [args[1], args[0], dy, dx, ystep, xstep]
                
            if len(args) == 5:
                match xstep + 4 * ystep:
                    case -5: pystep = -1; pxstep =  1
                    case -1: pystep = -1; pxstep =  0
                    case  3: pystep =  1; pxstep =  1
                    case -4: pystep =  0; pxstep = -1
                    case  0: pystep =  0; pxstep =  0
                    case  4: pystep =  0; pxstep =  1
                    case -3: pystep = -1; pxstep = -1
                    case  1: pystep = -1; pxstep =  0
                    case  5: pystep =  1; pxstep = -1
                if flip:
                    new_args.append(pystep)
                    new_args.append(pxstep)
                else:
                    new_args.append(pxstep)
                    new_args.append(pystep)
                new_args.append(args[4])
                new_args.append(flip)

            return np.flip(func(*new_args), axis=1) if flip else func(*new_args)

        return wrapper

    @_bresenline_wrapper
    @staticmethod
    def bresenline(x0, y0, dx, dy, xstep, ystep):
        yi, dy = (1, dy) if dy >= 0 else (-1, -dy)
        array = np.zeros((0, 2), dtype=int)
        D = 2 * dy - dx
        Dd = 2 * (dy - dx)
        Di = 2 * dy
        x = x0
        y = y0
        for x in range(x0, x0 + dx + 1):
            array = np.concatenate((array, [[x, y]]), axis=0)
            if D > 0:
                y += yi
                D += Dd
            else:
                D += Di
            x += xstep

        return array

    @_bresenline_wrapper
    @staticmethod
    def thick_bresenline(x0, y0, dx, dy, xstep, ystep, pxstep, pystep, width, yflip=False):
        def perpendicular(x0, y0, dx, dy, pxstep, pystep, einit, width, winit, yflip):
            array = np.zeros((0, 2), dtype=int)
            threshold = dx - 2 * dy
            E_diag = -2 * dx
            E_square = 2 * dy
            wthr = 2 * width * (dx**2 + dy**2) ** 0.5

            x, y = x0, y0
            error = -einit if yflip else einit
            tk = dx + dy + winit if yflip else dx + dy - winit
            p, q = 0, 0

            while tk <= wthr:
                array = np.concatenate((array, [[x, y]]), axis=0)
                if (error > threshold) or (error >= threshold and not yflip):
                    x += pxstep
                    error += E_diag
                    tk += 2 * dy
                error += E_square
                y += pystep
                tk += 2 * dx
                q += 1

            x, y = x0, y0
            error = einit if yflip else -einit
            tk = dx + dy - winit if yflip else dx + dy + winit

            while tk <= wthr:
                if p:
                    array = np.concatenate((array, [[x, y]]), axis=0)
                if (error > threshold) or (error >= threshold and yflip):
                    x -= pxstep
                    error += E_diag
                    tk += 2 * dy
                error += E_square
                y -= pystep
                tk += 2 * dx
                p += 1

            if q == 0 and p < 2:
                array = np.concatenate((array, [[x0, y0]]), axis=0)

            return array

        array = np.zeros((0, 2), dtype=int)
        p_error = 0
        error = 0
        x, y = x0, y0
        threshold = dx - 2 * dy
        E_diag = -2 * dx
        E_square = 2 * dy

        for p in range(dx + 1):
            array = np.concatenate(
                (array, perpendicular(x, y, dx, dy, pxstep, pystep, p_error, width, error, yflip)), axis=0
            )
            if error >= threshold:
                y += ystep
                error += E_diag
                if p_error >= threshold:
                    array = np.concatenate(
                        (
                            array,
                            perpendicular(x, y, dx, dy, pxstep, pystep, p_error + E_diag + E_square, width, error, yflip),
                        ),
                        axis=0,
                    )
                    p_error += E_diag
                p_error += E_square
            error += E_square
            x += xstep

        return array


class BasicSolverLeapfrog:
    """Solves leapfrog very basically, specifically for this."""

    def __init__(self, func, tstep, x0, v0):
        self.func = func
        self.tstep = tstep
        self.t = 0
        self.x = x0
        self.v = v0

    def __next__(self):
        a = self.func(self.x, self.t)
        self.x += self.v * self.tstep + 0.5 * a * self.tstep**2
        self.t += self.tstep
        self.v += 0.5 * (a + self.func(self.x, self.t)) * self.tstep
        return self.x, self.v


class PendulumGenerator:
    """Draws a pendulum. More or less."""

    def __init__(self, **kwargs):

        defaultKwargs = {
            "logger": None,
            "name": "pendulum",
            # Graphing constants
            # "tot_points": 1000000,
            # "new_line_weight": 0.0001,
            "im_width": 1920,
            "im_height": 1920,
            "dots_per_cm": 20,  # Chosen since max. length 43*2=86 cm, 96 cm divides 1920 nicely
            "time_step": 0.005,
            "output_name": "coolgif",
            "color1": (0, 0, 255),
            "color2": (255, 0, 0),
            "redo_gradient": False,
            "num_colors": 10000,
            "frames": 2000,
            # Physical constants
            "g_acc": 9.81,
            "wheel_radius": 3,
            "wheel_angular_speed": 10,
            "pendulum_length": 40,
            # Initial values
            "angle": -np.pi / 4,
        }

        kwargs = defaultKwargs | kwargs
        self.__dict__.update(kwargs)

        if self.logger is None:
            self.logger = logging.getLogger(f"{type(self).__name__} {self.name}")

        self.im_array = np.zeros((self.im_height, self.im_width, 3))
        self.pt_array = np.zeros((self.im_height, self.im_width))

        # self.time_step = 1 / (
        #     self.pendulum_length * self.dots_per_cm * self.wheel_angular_speed
        # )  # sin small angles approx. x
        self.gamma = self.g_acc / (self.pendulum_length * self.wheel_angular_speed**2)
        self.lambda_ = self.wheel_radius / self.pendulum_length

        self.wheel_center = (int(self.im_width // 2), int(self.im_height // 2))

        self.drawer = self.drawer(
            self.wheel_center, self.wheel_radius, self.pendulum_length, self.dots_per_cm
        )
        self.solver = self.solver(
            lambda phi, tau: -self.gamma * np.sin(phi) - self.lambda_ * np.sin(phi - tau),
            self.wheel_angular_speed * self.time_step,
            self.angle,
            0,
        )
        self.gradgen = self.gradgen(self.color1, self.color2)
        self.logger.info(f"Initializing color gradient.")
        start = time.perf_counter()
        if (not os.path.isfile("gradient.bytes")) or self.redo_gradient:
            self.logger.warning("Regenerating the gradient file.")
            self.colors = np.array(
                [self.gradgen.color(ratio) for ratio in np.linspace(0, 1, self.num_colors + 1)]
            )
            with open("gradient.bytes", "wb") as file:
                pickle.dump(self.colors, file)
        else:
            with open("gradient.bytes", "rb") as file:
                self.colors = pickle.load(file)
        self.logger.info(f"Finished gradient. Took {time.perf_counter()-start} s.")

        self.max_vals = []

        self.time = 0
        self.frame_count = 1

        self.logger.debug(f"{self.gamma=}; {self.lambda_=}; {self.wheel_center=}")

    def run(self):
        """
        Start the code for a chaotic pendulum concentration graph using vispy.

        Returns
        -------
        None.

        """
        big_start = time.perf_counter()
        start = time.perf_counter()
        line_points = self.drawer.line(self.angle, self.wheel_angular_speed * self.time, 4)
        self.circle_points, self.filled_circle = self.drawer.circle(5)
        self.circle_points = np.flip(self.circle_points.T, axis=0)
        self.filled_circle = np.flip(self.filled_circle.T, axis=0)
        self.logger.debug(
            f"Generated initial line and circle points. Took {time.perf_counter()-start}"
        )

        start = time.perf_counter()
        self.pt_array[*np.flip(line_points.T, axis=0)] += 1
        self.pt_array[*self.filled_circle] *= 0.5
        self.logger.debug(
            f"Completed initial point calculations. Took {time.perf_counter()-start}"
        )

        start = time.perf_counter()
        im_array = self.colors[
            (np.arctan(self.pt_array) * self.num_colors /
             (np.max(np.arctan(self.pt_array)))).astype(int)
        ]
        self.max_vals.append(np.median(self.pt_array[self.pt_array != 0]))
        # im_array = self.colors[
        #     (self.pt_array * self.num_colors / np.max(self.pt_array)).astype(int)
        # ]
        self.logger.debug(
            f"Completed colourization of initial image. Took {time.perf_counter()-start}"
        )

        start = time.perf_counter()
        im_array[self.pt_array == 0] = [0, 0, 0]
        im_array[*np.flip(line_points.T, axis=0), :] = [0, 255, 0]
        im_array[*self.circle_points, :] = [0, 255, 0]
        self.logger.debug(f"Completed drawing of initial frame in {time.perf_counter()-start}")

        start = big_start
        # with imageio.get_writer('coolgif.mp4', mode='I', fps=int(1/self.time_step)) as writer:
        with iio.imopen(self.output_name+".mkv", "w", plugin="pyav") as out_file:
            out_file.init_video_stream("vp9", fps=int(1/self.time_step))
            # writer.append_data(np.flip(im_array.astype(np.uint8), axis=0))
            out_file.write_frame(np.flip(im_array.astype(np.uint8), axis=0))
            for _ in range(self.frames - 1):
                self.logger.info(f"Finished frame {self.frame_count}/{self.frames} in {time.perf_counter()-big_start:.2f} s")
                self.frame_count += 1
                big_start = time.perf_counter()
                im_array = self._update_norm()
                # writer.append_data(np.flip(im_array.astype(np.uint8), axis=0))
                out_file.write_frame(np.flip(im_array.astype(np.uint8), axis=0))
            self.logger.info(f"Finished frame {self.frame_count}/{self.frames} in {time.perf_counter()-big_start:.2f} s")
        diff = time.perf_counter()-start
        self.logger.info(f"Whole process took {diff:.2f} s, or {diff/60:.2f} min, and had on average {self.frames/diff:.2f} fps or {diff/self.frames:.2f} spf")

    def _update_multi(self):
        """Update function with hopes for multiprocessing."""
        pass

    def _update_norm(self, ev=None):
        """Update function w/o multiprocessing."""
        start = time.perf_counter()
        self.time += self.time_step
        self.angle = next(self.solver)[0]
        line_points = self.drawer.line(self.angle, self.wheel_angular_speed * self.time, 4)
        self.logger.debug(
            f"Completed line calculations. Took {time.perf_counter()-start}"
        )

        start = time.perf_counter()
        self.pt_array[*np.flip(line_points.T, axis=0)] += 1
        self.pt_array[*self.filled_circle] *= 0.5
        self.logger.debug(f"Completed point calculations. Took {time.perf_counter()-start}")

        start = time.perf_counter()
        im_array = self.colors[
            (np.arctan(self.pt_array/(self.frame_count**0.55)) * self.num_colors /
             (np.max(np.arctan(self.pt_array/(self.frame_count**0.55))))).astype(int)
        ]
        self.max_vals.append(np.median(self.pt_array[self.pt_array != 0]))
        # im_array = self.colors[
        #     (self.pt_array * self.num_colors / np.max(self.pt_array)).astype(int)
        # ]
        self.logger.debug(f"Completed colouring. Took {time.perf_counter()-start}")

        start = time.perf_counter()
        im_array[self.pt_array == 0] = [0, 0, 0]
        im_array[*np.flip(line_points.T, axis=0), :] = [0, 255, 0]
        im_array[*self.circle_points, :] = [0, 255, 0]
        self.logger.debug(f"Completed drawing. Took {time.perf_counter()-start}")

        return im_array

    def test(self):
        mask = self.generate_line(-np.pi / 2, 0)
        print(mask)
        array_ = np.zeros((1920, 1920), dtype=int)
        array_[mask] = 1
        return array_, mask


if __name__ == "__main__":
    level = logging.INFO
    fmt = "[%(levelname)s|%(name)s] %(asctime)s: %(message)s"
    logging.basicConfig(stream=sys.stdout, level=level, format=fmt, datefmt="%Y-%m-%dT%H:%M:%S%z")

    pendulum = PendulumGenerator(
        angle=-1 * np.pi / 8,
        drawer=Drawer,
        solver=BasicSolverLeapfrog,
        gradgen=Gradient,
        output_name = "coolgif2",
        frames=5000,
    )
    pendulum.run()
