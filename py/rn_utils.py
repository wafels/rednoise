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

    fft_sim = wn_fft[1:] * f[1:]**(-alpha/2.0)
    T_sim = np.fft.irfft(fft_sim)
    return T_sim



def power_law_noise(n, dt, beta=1):
    """ Create a real valued time series with the given power 
    distribution f**-beta.  Follows Timmers and Konig 1995, A. &. A., 300, 707.
    
    Parameters
    -----------
    n: integer
        Length of time-series

    dt: float
        sample cadence of the time series

    
    """
    pass
    return None