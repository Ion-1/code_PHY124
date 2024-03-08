# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 13:52:37 2024

@author: Ion-1
"""

def palindrome_checker(): return (lambda x: x == x[::-1])("".join([char.lower() for char in input("Input a string:\n") if char not in [" ", "\n", "_"]]))