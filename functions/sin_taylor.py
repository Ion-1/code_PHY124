# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:46:49 2024

@author: Ion-1
"""
import numpy as np


def taylor_sin(x, eps=1e-8):
    
    counter = 1
    sum_x = x
    term = x
    
    while abs(term) > 0.5*eps:
        
        counter += 2
        
        term *= (-1)*(x**2)/(counter*(counter-1))
        
        sum_x += term
        
    return sum_x


print(f"{taylor_sin(-3*np.pi/14):.8f}")