# -*- coding: utf-8 -*-
"""
Created on Tue May 14 09:29:31 2024.

@author: Ion-1
"""
import numpy as np


if __name__=="__main__":
    m = 1/0.26
    r = 1+1+1/0.26
    A = np.mat([[r, -m, -1, 0, -1, 0, 0, 0],
                [-m, r, 0, -1, 0, -1, 0, 0],
                [-1, 0, 3, -1, 0, 0, -1, 0],
                [0, -1, -1, 3, 0, 0, 0, -1],
                [-1, 0, 0, 0, 3, -1, -1, 0],
                [0, -1, 0, 0, -1, 3, 0, -1],
                [0, 0, -1, 0, -1, 0, 3, -1],
                [0, 0, 0, -1, 0, -1, -1, 3]])
    b = np.array([1, 0, 0, 0, 0, 0, 0, -1])
    V = np.linalg.solve(A, b)
    print(V[0]-V[7])
