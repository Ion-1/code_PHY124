# -*- coding: utf-8 -*-
"""
Created on Tue May 14 09:23:53 2024.

@author: Ion-1
"""
import numpy as np


if __name__=="__main__":
    A = np.mat([[2, -1, -1, 0],[-1,3,-1,-1],[-1,-1,3,-1],[0,-1,-1,2]])
    b = np.array([1,0,0,-1])
    ans = np.linalg.solve(A, b)
    print(ans[0]-ans[3])