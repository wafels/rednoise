"""
Spectra for use with the red noise MCMC analysis
"""

import numpy as np

def power_law(f, a):
    return np.exp(a[0]) * (f ** (-a[1]))

def power_law_with_constant(f, a):
    return np.exp(a[0]) * (f ** (-a[1])) + np.exp(a[2])

def broken_power_law(f, a):
    f0 = np.exp(a[3])
    A = np.exp(a[0])
    out = np.zeros(shape=f.shape)
    out[f<f0] = A * (f[f<f0] ** (-a[1]))
    out[f>=f0] = A * (f[f>=f0] ** (-a[2])) * (f0 ** (a[2] - a[1]))
    return out