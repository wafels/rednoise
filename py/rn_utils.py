"""
Utility functions for the 1/f study
"""

import numpy as np


def power_law_noise(n, dt, alpha):
    """Create a time series with power law noise"""

    # White noise
    wn = np.random.normal(size=(n))

    # FFT of the white noise - chi2(2) distribution
    wn_fft = np.fft.rfft(wn)

    # frequencies
    f = np.fft.fftfreq(n, dt)[:len(wn_fft)]
    f[-1] = np.abs(f[-1])

    fft_sim = wn_fft[1:] * f[1:] ** (-alpha / 2.0)
    T_sim = np.fft.irfft(fft_sim)
    return T_sim


def simulated_power_law(n, dt, alpha, n_oversample=10, dt_oversample=10):
    """Create a time series of length n and sample size dt"""
    data = power_law_noise(n_oversample * n, dt / dt_oversample, alpha)
    data = data - np.mean(data)
    return data[0.5 * len(data) - n / 2: 0.5 * len(data) + n / 2]
