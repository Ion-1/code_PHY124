# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 14:09:53 2024

@author: Ion-1
"""

def four_cubes():
    
    d = 1
    
    while True:
        
        d += 1
        cube = d**3
        
        for c in range(1,d):
            for b in range(1, c+1):
                for a in range(1, b+1):
                    if a**3 + b**3 + c**3 == cube:
                        return a, b, c, d

if __name__ == "__main__":
    print(four_cubes())