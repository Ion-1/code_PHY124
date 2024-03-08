# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 13:46:13 2024

@author: Ion-1
"""

def hypergeo_func(a,b,c,z,n):
    
    sum_ = 1
    term = 1
    k = 1
    
    while abs(term)*10**(n+1) > 1:
        
        term *= (a+(k-1))*(b+(k-1))*z/((c+(k-1))*k)
        sum_ += term
        k += 1
    
    return round(sum_, n)

if __name__ == "__main__":
    print(hypergeo_func(1, 1, 2, -0.87388, 6))