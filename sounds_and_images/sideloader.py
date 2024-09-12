# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 00:49:39 2024

@author: Ion-1
"""
import numpy as np

class TI_Interface:
    
    def __init__(self, origin, t_mtrx, init, logger):
        self.origin = origin
        self.t_mtrx = t_mtrx
        self.init = init
        self.logger = logger
    
    def __call__(self, coords):
        x, y = coords
        # self.logger.info(f"Calculating color for ({x}, {y})")
        init_x, init_y = self.init
        from_origin = np.array([x, y]) - np.array([init_x, init_y])
        derotate = np.matmul(np.linalg.inv(self.t_mtrx), from_origin.T).T
        re_add_origin = derotate + self.origin
        prev_x, prev_y = int(re_add_origin[0]), int(re_add_origin[1])
        if 0 <= prev_x < self.origin[0]*2 and 0 <= prev_y < self.origin[1]*2:
            return (x, y, (prev_x, prev_y))
        return (x, y, (None, None))

def basic_function(c):
    return (c[0],c[1],(c[0]//2, c[1]//2))