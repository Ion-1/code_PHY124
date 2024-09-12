# -*- coding: utf-8 -*-
"""
Created on Fri May 24 15:04:01 2024.

@author: Ion-1
"""
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation

from sklearn.utils import Bunch


def quantum_leapfrog(
    v_func: callable,
    y0: cp.ndarray,
    t_range: tuple,
    del_x: float,
    del_t: float,
) -> Bunch:
    """
    Leapfrog algorithm for the wave function

    Parameters
    ----------
    v_func : callable
        DESCRIPTION.
    y0 : cp.ndarray
        DESCRIPTION.
    t_range : tuple
        DESCRIPTION.
    del_t : float
        DESCRIPTION.
    del_x : float
        DESCRIPTION.

    Returns
    -------
    Bunch
        DESCRIPTION.

    """
    y = []
    p = []
    N = len(y0)
    del_k = 2 * cp.pi / N / del_x
    n = cp.arange(0, N) - N // 2
    x = del_x * n
    v_x = v_func(x)
    k = del_k * n
    k_drift = cp.exp(-1j * k**2 * del_t / 4)
    v_kick = cp.exp(-1j * v_x * del_t)
    t = cp.arange(t_range[0], t_range[1] + del_t, del_t)
    working_array = y0.astype(cp.complex128)
    for _ in t:
        working_array = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(working_array)))
        working_array = k_drift * working_array
        working_array = cp.fft.fftshift(cp.fft.ifft(cp.fft.fftshift(working_array)))
        working_array = v_kick * working_array
        working_array = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(working_array)))
        working_array = k_drift * working_array
        p.append(working_array.get())
        working_array = cp.fft.fftshift(cp.fft.ifft(cp.fft.fftshift(working_array)))
        y.append(working_array.get())
    return Bunch(y=y, t=t, p=p)


if __name__ == "__main__":

    fig, (ax1, ax2) = plt.subplots(2, 1)

    del_x = 0.2
    st, en = -4, 4
    x = cp.arange(st, en, del_x)
    ground = cp.exp(-x**2/2)
    # ground = cp.array([0] * 85 + [1] * 30 + [0] * 85)
    result = quantum_leapfrog(
        lambda y: 0.5 * y**2, ground, (0, 100), del_x, 0.1
    )

    x = x.get()
    artists = []
    for phi, hat in zip(result.y, result.p):
        batch = []
        batch = batch + ax1.plot(x, abs(phi)**2, color='black')
        batch = batch + ax1.plot(x, np.real(phi), color='blue')
        batch = batch + ax1.plot(x, np.imag(phi), color='orange')
        batch = batch + ax2.plot(x, abs(hat), color='black')
        batch = batch + ax2.plot(x, np.real(hat), color='blue')
        batch = batch + ax2.plot(x, np.imag(hat), color='orange')
        artists.append(batch)
    ani = ArtistAnimation(fig=fig, artists=artists, interval=20, repeat=False)
