# -*- coding: utf-8 -*-
"""
Created on Mon May 20 23:57:05 2024.

@author: Ion-1
"""
import numpy as np
from morsecode import MORSE


def decoder(signal: str, MORSE) -> str:
    return "".join([eval("MORSE"+".get('children')".join([f".get('{char}')" for char in morse])+".get('label')") for morse in signal.split()])

def functional_decoder(signal, MORSE):
    return "".join(map(lambda morse: eval("MORSE"+".get('children')".join(map(lambda char: f".get('{char}')", morse))+".get('label')"), signal.split()))


if __name__ == "__main__":
    print(decoder('--.. . .-. ---', MORSE))
    print((lambda signal: "".join([eval("MORSE"+".get('children')".join([f".get('{char}')" for char in morse])+".get('label')") for morse in signal.split()]))('--.. . .-. ---'))
    print(functional_decoder('--.. . .-. ---', MORSE))
    print((lambda signal:"".join(map(lambda morse: eval("MORSE"+".get('children')".join(map(lambda char: f".get('{char}')", morse))+".get('label')"), signal.split())))('--.. . .-. ---'))