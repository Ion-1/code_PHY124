# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 14:13:34 2024.

@author: Ion-1
"""
import os

file_name = "genetic_code.txt"
current_wdir = os.getcwd()

path = file_name if os.path.isabs(file_name) else os.path.join(current_wdir,file_name)
if not os.path.exists(path): raise FileNotFoundError(f'path {path} does not exist')

with open(path, 'r', newline='') as file:
    
    reader = file.readlines()
    
    genetic_code = {}
    
    for line in reader:
        
        code, acid = line.rstrip('\n').split(' ', maxsplit=1)
        
        if acid in genetic_code.keys():
            genetic_code[acid].append(code)
        else:
            genetic_code[acid] = [code]

genetic_code['Leucine'].sort()
print(genetic_code['Leucine'][1])