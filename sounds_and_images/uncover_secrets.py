# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 16:14:21 2024

@author: Ion-1
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
import logging


level = logging.WARNING
fmt = "[%(levelname)s] %(asctime)s: %(message)s"
logging.basicConfig(level=level, format=fmt)


with Image.open("36.png") as image:
    data = np.array(image)
# data = plt.imread("36.png")

start = time.perf_counter()
array1 = np.array(
    [
        [0.01 if y in range(100 * x, 100 * (x + 1)) else 0 for y in range(1000)]
        for x in range(10)
    ]
)
array2 = np.array(
    [
        [0.01 if x in range(100 * y, 100 * (y + 1)) else 0 for y in range(30)]
        for x in range(3000)
    ]
)

end = time.perf_counter()
print(f"Matrix Creation M1: {end-start}")

def func(shape, flat=True):
    array = np.zeros(shape)
    with np.nditer(array, flags=['multi_index'], op_flags=['writeonly']) as it:
        for item in it:
            x, y = it.multi_index[0], it.multi_index[1]
            if flat: item[...] = 0.01 if 100*x <= y and y < 100*(x+1) else 0
            else: item[...] = 0.01 if 100*y <= x and x < 100*(y+1) else 0
    return array
array1 = func((10,1000), True)
array2 = func((3000,30), False)
start = time.perf_counter()
print(f"Matrix Creation M2: {start-end}")
new_im = np.matmul(
        np.matmul(
            array1,
            data,
        ),
        array2,
    )
end = time.perf_counter()
print(f"Matrix mult: {end-start}s")
plt.imshow(new_im)
new_im = Image.fromarray(new_im)

start = time.perf_counter()
new_im_2 = np.stack(
        [
            arr.mean(axis=1)
            for arr in np.hsplit(
                np.stack([array.mean(axis=0) for array in np.vsplit(data, 10)]), 30
            )
        ]
    ).T
end = time.perf_counter()
print(f"Array chopping: {end-start}s")

start = time.perf_counter()
new_im_3 = np.zeros((10,30))
for i in range(10):
    for j in range(30):
        new_im_3[i][j] = np.mean(data[100*i:100*(i+1), 100*j:100*(j+1)])
end = time.perf_counter()
print(f"Loops: {end-start}s")

# new_im.save("37.png", "PNG")
# # new_im_2.save("38.png", "PNG")
# new_im_3.save("39.png", "PNG")

# plt.imshow(new_im)
# plt.imshow(new_im_3)
