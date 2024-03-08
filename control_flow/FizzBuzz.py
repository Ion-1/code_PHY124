# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 14:08:19 2024

@author: Ion-1
"""
def fizz_buzz_bad(start=1, end=15):
    
    string = ""
    
    n = start
    
    while n <= end:
        
        if n % 15 == 0:
            string = string + "fizzbuzz"
        elif n % 3 == 0:
            string = string + "fizz"
        elif n % 5 == 0:
            string = string + "buzz"
        else:
            string = string + f"{n}"
        
        if n != end:
            string = string + ", "
            
        n += 1
    
    return string