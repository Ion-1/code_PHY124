# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 15:03:38 2024.

@author: Ion-1
"""

import numpy as np
from itertools import chain
import matplotlib.pyplot as plt

filename = "dice.txt"

with open(filename, 'r') as file:
    array = list(map(lambda x: x[0]+x[1]+x[2], np.loadtxt(file,
                 dtype=int, delimiter=' ', converters=lambda x: int(x))))

distribution = [0]*16
for i in array:
    distribution[i-3] += 1

plt.style.use("seaborn-v0_8-dark")

fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.hist(array, bins=[i-0.5 for i in range(3, 20)])
ax2.bar(list(range(3,19)), distribution, width=1, fill=True, xerr=0.5, yerr=list(map(
    lambda x: np.sqrt(x), distribution)), capsize=4, error_kw={"lw": 2, "capthick": 2}, ec="#eaeaf2", lw=1)
ax2.plot(list(range(3,19)), distribution, "ko-", lw=3)

fig.suptitle("Histograms, but one is a bar in disguise")

ax1.set_xlabel("Dice roll", loc='center',
               fontstyle='oblique', fontsize='medium')
ax1.set_ylabel("Entries", loc='center',
               fontstyle='oblique', fontsize='medium')
ax1.set_title("A histogram", loc='center', fontstyle='oblique',
              fontsize='medium', pad=10)
ax1.set_xticks(list(range(3,19)))

ax2.set_xlabel("Dice roll", loc='center',
               fontstyle='oblique', fontsize='medium')
ax2.set_ylabel("Entries", loc='center',
               fontstyle='oblique', fontsize='medium')
ax2.set_title("A bar, but NO alcohol", loc='center', fontstyle='oblique',
              fontsize='medium', pad=10)
ax2.set_xticks(list(range(3,19)))

x = np.linspace(2, 20, 1000)
std = np.std(array)
y = list(map(lambda x: 10000*np.exp(-0.5*((x-10.5)/std)**2)/(std*np.sqrt(2*np.pi)), x))

ax1.plot(x, y, "r--", lw=3)
ax2.plot(x, y, "r--",lw=3)

plt.show()
