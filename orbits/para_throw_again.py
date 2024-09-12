# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 10:13:45 2024

@author: Ion-1
"""
from scipy.integrate import solve_ivp, odeint
import numpy as np

def two_ode(t, y): 
    return [y[2], y[3], 0, -9.81]

if __name__ == "__main__":
    print(solve_ivp(two_ode, (0, 1.3), [0, 0, 11, 7]))
    print(odeint(two_ode, [0, 0, 11, 7], np.linspace(0, 1.3, 100), tfirst=True))
