import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift


def fourier(f):
    """fourier transform"""
    return fftshift(fft2(fftshift(f)))


def ifourier(f):
    """inverse fourier transform"""
    return fftshift(ifft2(fftshift(f)))


# any png-file should work here
src = plt.imread("example.png")[:, :, :3]
plt.imshow(src)

Y, X, _ = src.shape
x = (np.arange(X) - X // 2) * 2 * np.pi / X
y = (np.arange(Y) - Y // 2) * 2 * np.pi / Y
kx, ky = np.meshgrid(x, y)

f = np.zeros_like(src, dtype=complex)
img = np.zeros_like(src)

for t in range(1000):
    for c in range(3):
        f[:, :, c] = fourier(src[:, :, c])
        f[:, :, c] *= np.exp(2 * (np.cos(kx) + np.cos(ky) - 2) * t)
        img[:, :, c] = ifourier(f[:, :, c]).real
    img = img.clip(0, 1)
    plt.cla()
    # it might be instructive to look at the fourier space as well
    # plt.imshow(f.real)
    plt.imshow(img)
    plt.pause(0.05)
