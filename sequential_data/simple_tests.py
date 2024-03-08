# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 14:22:22 2024

@author: Ion-1
"""
import numpy as np
import small_library as sh

if __name__ == "__main__":
    print(sh.filter_(np.array([8, 9, 3, 1, 2, 2, 2, 2, 5, 4, 9]), [8]))
    print(sh.find((7, 3, 9, 9, 7, 1, 5, 3, 4, 8, 1), 8))
    print(sh.find([0.3, 7.4, 9.8, 2.6, 6.0, 7.4, 4.3, 9.2, 3.2, 9.8], 9.8))
    print(sh.common([7, 3, 9, 9, 7, 1, 5, 3, 4, 8, 1],
          [8, 9, 3, 1, 2, 2, 2, 2, 5, 4, 9]))

    l1 = [95, 96, 68, 70, 87, 59, ' ', 45, 29, 36, ' ',
          13, 38, 62, 65, 12, 80, ' ', 31, 49, 61, 53, 58]
    l2 = [57, 62, 59, 31, ' ', 14, 41, ' ', 90, 98, 52,
          17, 44, 67, 11, 55, 12, 91, 96, 64, 50, ' ', 27]
    print((lambda x: (sh.find(l1, x), sh.find(l2, x)))(
        max(sh.common(sh.filter_(l1, [' ']), sh.filter_(l2, [' '])))))