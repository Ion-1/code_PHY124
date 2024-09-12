# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 10:13:24 2024

@author: Ion-1
"""

import os
# os.environ['MKL_NUM_THREADS'] = '4'
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import os.path as pth
import mimetypes
import logging
import time
import concurrent.futures
from os import getpid
import itertools as itert
import sys

# with Image.open("holbein.png") as img:
#       data = np.swapaxes(np.array(img), 0, 1)/255

# theta = -15 * np.pi / 128
# h_stretch = 1
# v_stretch = 5
# origin = np.array(data.shape[:2])/2

# t_matrix = np.array([[h_stretch*np.cos(theta), h_stretch*np.sin(theta)],[-v_stretch*np.sin(theta), v_stretch*np.cos(theta)]])

# points = np.moveaxis(np.indices(data.shape[:2]), 0, -1).reshape(-1,2)
# t_points = np.matmul(t_matrix, (points-origin.reshape(1,2)).T).T + np.array([0.5*h_stretch*np.sqrt(1039**2+1024**2)*np.sin(np.arctan(1039/1024)+abs(theta)), 0.5*v_stretch*np.sqrt(1039**2+1024**2)*np.sin(np.arctan(1024/1039)+abs(theta))]).reshape(1,2)

# image_shape = (int(max(t_points[:,0])+1),int( max(t_points[:,1])+1))

# image = np.zeros((*image_shape, 3))
# normalize = np.zeros(image_shape, dtype=int)

# for i in range(t_points.shape[0]):
#     item = t_points[i]
#     x, y = int(item[0]), int(item[1])
#     image[x,y] *= normalize[x,y]
#     image[x,y] += data[*points[i]]
#     normalize[x,y] += 1
#     image[x,y] /= normalize[x,y]

# plt.imshow(image)
class Worker:
    def __init__(self, data):
        self.data = data
    def __call__(self, p):
        if (0 <= p[0] <= self.data.shape[0] - 1) and (
            0 <= p[1] <= self.data.shape[1] - 1
        ):
            p, w = np.divmod(p, 1)
            p = p.astype(int, copy=False)
            try:
                return (
                    (
                        self.data[p[0], p[1]] * (1 - w[0]) * (1 - w[1])
                        + self.data[p[0] + 1, p[1]] * w[0] * (1 - w[1])
                        + self.data[p[0], p[1] + 1] * (1 - w[0]) * w[1]
                        + self.data[p[0] + 1, p[1] + 1] * w[0] * w[1]
                        if p[1] != self.data.shape[1] - 1
                        else self.data[p[0] + 1, p[1]] * (1 - w[1])
                        + self.data[p[0], p[1]] * (1 - w[0])
                    )
                    if p[0] != self.data.shape[0] - 1
                    else (
                        self.data[p[0], p[1]] * (1 - w[1]) + self.data[p[0], p[1] + 1] * w[1]
                        if p[1] != self.data.shape[1] - 1
                        else self.data[p[0], p[1]]
                    )
                )
            except IndexError:
                print(f"IndexError {p=} {w=}")
        else:
            return np.array([0,0,0])

class prelim_filter:
    def __init__(self, diag, init, shape, rot):
        self.diag = diag
        self.init = init
        self.shape = shape
        self.rot = rot
    def __call__(self, index): # Needs fixing 4 more rot
        x, y = index
        x_bot = self.init[0] - 0.5*self.diag*np.sin(np.arctan(self.shape[0]/self.shape[1]) + self.rot) # x_bot is on top in the graph, just because
        x_top = 2*self.init[0] - x_bot
        y_left = self.init[1] + 0.5*self.diag*np.sin(np.arctan(self.shape[1]/self.shape[0]) + self.rot)
        y_right = 2*self.init[1] - y_left
        return (y >= y_right*(x-x_bot)/x_top if x > x_bot else y >= y_left - y_left*x/x_bot) and (y <= 2*self.init[1]-y_left*(x-x_top)/x_bot if x > x_top else y <= y_left + y_right*x/x_top)

class TransformedImage:

    def __init__(
        self,
        data_or_fp: np.ndarray or str,
        rotation: int or float = 0,
        stretch_factor: tuple = (1, 1),
        logger=None,
    ):
        self.logger = logger

        if isinstance(data_or_fp, np.ndarray):
            if len(data_or_fp.shape) not in [2, 3]:
                raise ValueError("The array must be at least 2-dimensional and at maximum 3.")
            self.data = np.swapaxes(data_or_fp, 0, 1)  # swap x-, y-axis to prevent insanity
        elif isinstance(data_or_fp, str):
            data_or_fp = pth.realpath(data_or_fp)
            ext = "image" in mimetypes.guess_type(data_or_fp)[0]
            exists = pth.isfile(data_or_fp)
            if not (exists and ext):
                if not exists:
                    raise FileNotFoundError("The given filepath does not point to a existing file")
                raise TypeError("Either your filetype is weird or wrong.")
            with Image.open(data_or_fp) as img:
                self.data = (
                    np.swapaxes(np.array(img), 0, 1) / 255
                )  # swap x-, y-axis to prevent insanity
        else:
            raise TypeError("Your data input is not an array or a filepath.")

        self.shape = np.array(self.data.shape[:2])

        self.stretch = np.array([[stretch_factor[0], 0], [0, stretch_factor[1]]])
        self.rot = rotation if isinstance(rotation, float) else np.pi * rotation / 180
        self.rotation = np.array(  # Still a counter-clockwise rotation due to downwards y-axis
            [
                [np.cos(self.rot), np.sin(self.rot)],
                [-np.sin(self.rot), np.cos(self.rot)],
            ]
        )

    def generate_array(self):
        diag = np.sqrt(np.sum(np.array(self.shape) ** 2))
        init = self.stretch @ np.array(
            [
                0.5 * diag * abs(np.sin(np.arctan(self.shape[0] / self.shape[1]) + abs(self.rot))),
                0.5 * diag * abs(np.sin(np.arctan(self.shape[1] / self.shape[0]) + abs(self.rot))),
            ]
        )
        init = init.astype(int, copy=False)
        array = np.zeros((int(2 * init[0]), int(2 * init[1]), 3))
        print(
            f"Expected number of entries 2 go: {array.shape[0]*array.shape[1]/1000000:.1f} million"
        )

        t_matrix = np.linalg.inv(self.stretch @ self.rotation)

        bef_time = 0
        tot_time = 0
        cal_time = 0
        ass_time = 0
        prev_i = 0

        big_start = time.perf_counter()
        for i, ((x, y), p) in enumerate(
            zip(
                np.ndindex(array.shape[:2]),
                np.matmul(
                    t_matrix,
                    (np.array(list(np.ndindex(array.shape[:2]))) - init.reshape(1, 2)).T,
                ).T
                + self.shape / 2,
            )
        ):
            before = time.perf_counter()
            if int(i / 100000) != prev_i:
                print(f"Finished {i/1000000:_} millionth entry!")
                prev_i = int(i / 100000)
            if (0 <= p[0] <= self.shape[0] - 1) and (
                0 <= p[1] <= self.shape[1] - 1
            ):  # np.all takes too long
                start = time.perf_counter()
                p, w = np.divmod(p, 1)
                p = p.astype(int, copy=False)
                mid = time.perf_counter()
                try:
                    array[x, y] = (  # Interpolation but edges are dangerous
                        (
                            self.data[p[0], p[1]] * (1 - w[0]) * (1 - w[1])
                            + self.data[p[0] + 1, p[1]] * w[0] * (1 - w[1])
                            + self.data[p[0], p[1] + 1] * (1 - w[0]) * w[1]
                            + self.data[p[0] + 1, p[1] + 1] * w[0] * w[1]
                            if p[1] != self.shape[1] - 1
                            else self.data[p[0] + 1, p[1]] * (1 - w[1])
                            + self.data[p[0], p[1]] * (1 - w[0])
                        )
                        if p[0] != self.shape[0] - 1
                        else (
                            self.data[p[0], p[1]] * (1 - w[1]) + self.data[p[0], p[1] + 1] * w[1]
                            if p[1] != self.shape[1] - 1
                            else self.data[p[0], p[1]]
                        )
                    )
                except IndexError:
                    print(f"IndexError ({x}, {y}) {p=} {w=}")
                end = time.perf_counter()
                tot_time += end - before
                cal_time += mid - start
                ass_time += end - mid
                bef_time += start - before
        mat_time = time.perf_counter() - big_start - tot_time
        print(f"{mat_time=}; {tot_time=}; {bef_time=}; {cal_time=}; {ass_time=}")

        array = np.swapaxes(array, 0, 1)

        return array

    def generate_array_par(self, max_workers=None):
        diag = np.sqrt(np.sum(np.array(self.shape) ** 2))
        init = self.stretch @ np.array(
            [
                0.5 * diag * abs(np.sin(np.arctan(self.shape[0] / self.shape[1]) + abs(self.rot))),
                0.5 * diag * abs(np.sin(np.arctan(self.shape[1] / self.shape[0]) + abs(self.rot))),
            ]
        )
        init = init.astype(int, copy=False)
        array = np.zeros((int(2 * init[0]), int(2 * init[1])), dtype=np.ndarray)
        new_shape = (int(2 * init[0]), int(2 * init[1]), 3)
        print(
            f"Expected number of entries 2 go: {array.shape[0]*array.shape[1]/1000000:.1f} million"
        )

        t_matrix = np.linalg.inv(self.stretch @ self.rotation)
        time_2 = 0

        array_list = []
        begin = time.perf_counter()
        with Pool(max_workers) as pool:
            for p in pool.imap(Worker(self.data), np.matmul(
                        t_matrix,
                        (np.array(list(np.ndindex(new_shape[:2]))) - init.reshape(1, 2)).T,
                    ).T
                    + self.shape / 2, chunksize=80000):
                begin_2 = time.perf_counter()
                array_list.append(p)
                # np.put_along_axis(array, ind, p, axis)
                time_2 += time.perf_counter()-begin_2
        time_1 = time.perf_counter() - begin - time_2
        print(f"debuf {time_1=} {time_2=}")

        array = np.array(array_list).reshape(new_shape)
        array = np.swapaxes(array, 0, 1)

        return array
    
    def generate_array_par_bad(self, max_workers=None):
        diag = np.sqrt(np.sum(np.array(self.shape) ** 2))
        init = self.stretch @ np.array(
            [
                0.5 * diag * abs(np.sin(np.arctan(self.shape[0] / self.shape[1]) + abs(self.rot))),
                0.5 * diag * abs(np.sin(np.arctan(self.shape[1] / self.shape[0]) + abs(self.rot))),
            ]
        )
        init = init.astype(int, copy=False)
        array = np.zeros((int(2 * init[0]), int(2 * init[1])), dtype=np.ndarray)
        new_shape = (int(2 * init[0]), int(2 * init[1]), 3)
        array_list = []

        print("Creating filter and generating mask")
        start = time.perf_counter()
        filter_ = prelim_filter(diag,init,self.shape,self.rot)
        with Pool(max_workers) as pool:
            indices_list = np.ma.make_mask(pool.map(filter_, np.ndindex(new_shape[:2]), chunksize=150000))
        print(f"Time taken {time.perf_counter()-start}")
        print(
            f"Expected number of entries 2 go: {indices_list.sum()/1000000:.1f} million"
        )

        t_matrix = np.linalg.inv(self.stretch @ self.rotation)
        time_2 = 0

        begin = time.perf_counter()
        with Pool(max_workers) as pool:
            for p in pool.imap(Worker(self.data), np.matmul(
                        t_matrix,
                        (np.array(list(np.ndindex(new_shape[:2])))[indices_list] - init.reshape(1, 2)).T,
                    ).T
                    + self.shape / 2, chunksize=80000):
                begin_2 = time.perf_counter()
                array_list.append(p)
                # np.put_along_axis(array, ind, p, axis)
                time_2 += time.perf_counter()-begin_2
        time_1 = time.perf_counter() - begin - time_2
        print(f"debuf {time_1=} {time_2=}")

        print("Reapplying mask")
        start = time.perf_counter()
        array = []
        i = iter(range(len(array_list)))
        for mask in indices_list.tolist():
            if mask:
                array.append(array_list[next(i)])
            else:
                array.append([0,0,0])
            
        print(f"Looping took {time.perf_counter()-start}")
                

        array = np.array(array).reshape(new_shape)
        array = np.swapaxes(array, 0, 1)

        return array


if __name__ == "__main__":
    plt.clf()
    plt.rcParams["figure.figsize"] = (18, 30)

    level = logging.INFO
    fmt = "[%(levelname)s] %(asctime)s: %(message)s"
    logging.basicConfig(stream=sys.stdout, level=level, format=fmt)
    logger = logging.getLogger(__name__)

    theta = -15 * np.pi / 128
    h_stretch = 1
    v_stretch = 7

    image = TransformedImage("holbein.png", theta, (h_stretch, v_stretch), logger)

    start = time.perf_counter()
    im_array = image.generate_array_par(10)
    im_array = image.generate_array_par_bad(12)
    print(f"{time.perf_counter()-start} seconds to generate and display array")
    plt.imshow(im_array)

# dorian.quelle@math.uzh.ch
