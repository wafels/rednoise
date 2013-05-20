"""
Utility functions for the 1/f study
"""

import numpy as np
import matplotlib.pyplot as plt


def power_law_noise(n, dt, alpha, seed=None):
    """Create a time series with power law noise"""

    # White noise
    np.random.seed(seed=seed)
    wn = np.random.normal(size=(n))

    # FFT of the white noise - chi2(2) distribution
    wn_fft = np.fft.rfft(wn)

    # frequencies
    f = np.fft.fftfreq(n, dt)[:len(wn_fft)]
    f[-1] = np.abs(f[-1])

    fft_sim = wn_fft[1:] * f[1:] ** (-alpha / 2.0)
    T_sim = np.fft.irfft(fft_sim)
    return T_sim


def simulated_power_law(n, dt, alpha, n_oversample=10, dt_oversample=10,
                        seed=None):
    """Create a time series of length n and sample size dt"""
    data = power_law_noise(n_oversample * n, dt / dt_oversample, alpha,
                           seed=seed)
    data = data - np.mean(data)
    return data[0.5 * len(data) - n / 2: 0.5 * len(data) + n / 2]


def credible_interval(data, ci=0.68):
    """find the limits where the upper 100*(1-ci/2)% of the data, and the lower
    100*ci/2 % of the data are"""
    s = np.sort(data)
    ns = s.size
    for i in range(0, ns):
        fraction = np.count_nonzero(s[i] > s) / (1.0 * ns)
        if fraction >= (1.0 - ci) / 2.0:
            break
    lower_limit = s[i]

    for i in range(ns - 1, 0, -1):
        fraction = np.count_nonzero(s[i] > s) / (1.0 * ns)
        if fraction <= 1.0 - (1.0 - ci) / 2.0:
            break
    upper_limit = s[i]
    return np.asarray([lower_limit, upper_limit])


def do_simple_fit(freq, power, show=False):
    """Do a very simple fit to the frequencies and power"""
    x = np.log(freq)
    y = np.log(power)
    coefficients = np.polyfit(x, y, 1)
    m_estimate = -coefficients[0]
    c_estimate = np.exp(coefficients[1])

    if show:
        # Fit to the power spectrum
        power_fit = c_estimate * freq ** (-m_estimate)

        # plot the power spectrum and the quick fit
        plt.figure(2)
        plt.loglog(freq, power, label='observed power')
        plt.loglog(freq, power_fit, label='fit, power law index: ' + str(m_estimate))
        plt.xlabel('frequency')
        plt.ylabel('power')
        plt.title('Data fit with simple single power law')
        plt.legend()
        plt.show()
    return [c_estimate, m_estimate]
