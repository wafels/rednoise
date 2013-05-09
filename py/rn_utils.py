"""
1/f noise utility functions
"""

import numpy as np


def power_law_noise1(n, dt, alpha):
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
