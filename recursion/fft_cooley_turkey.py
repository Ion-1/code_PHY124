# -*- coding: utf-8 -*-
"""
Created on Thu May 23 14:01:39 2024.

@author: Ion-1
"""
import numpy as np
import timeit
import matplotlib.pyplot as plt


def fft_recursive(s):
    if len(s) == 1:
        return s[0]
    H = len(s)//2
    s_loc = np.array(s)
    a_even = s_loc[::2]
    a_odd = s_loc[1::2]
    s_even = fft_recursive(a_even)
    s_odd = np.exp(-1j*np.pi*np.arange(0, H)/H)*fft_recursive(a_odd)
    return np.concatenate((s_even+s_odd, s_even-s_odd))


def fft_iterative(s):
    N = len(s)
    H = 1
    bigH = 2
    arr = np.array(s).astype(np.complex_)
    while bigH <= N:
        num = int(N/bigH)
        for n in range(num):
            s_even = arr[n::2*num]
            s_odd = np.exp(-1j*np.pi*np.arange(0, H)/H)*arr[n+num::2*num]
            comb = np.concatenate((s_even+s_odd, s_even-s_odd))
            arr[n::num] = comb
        bigH *= 2
        H *= 2
    return arr


if __name__ == "__main__":
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ini_vals = [5, 1, 1, 2, 9, 8, 1204, 8]
    result1 = fft_recursive(ini_vals)
    result2 = fft_iterative(ini_vals)
    print(result1[-1])
    print(result1)
    print(result2[-1])
    print(result2)
    x_vals = []
    timings_r = []
    timings_i = []
    for i in range(1, 17):
        a = timeit.timeit(lambda: fft_recursive(ini_vals*2**(i)), number=1)
        b = timeit.timeit(lambda: fft_iterative(ini_vals * 2 ** (i)), number=1)
        print(f"Recursive with list size {2**(i+3)}:", a)
        print(f"Numpy with list size {2**(i+3)}:", b)
        x_vals.append(2**(i+3))
        timings_r.append(a)
        timings_i.append(b)
    ax1.plot(x_vals, timings_r)
    ax2.plot(x_vals, timings_i)
