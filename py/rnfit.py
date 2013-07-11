"""
Given a time series, fit the given PyMC model to it
"""
import numpy as np
import pymc

def dopymc(PyMCmodel, data, dt, **kwargs):
    # Do the PYMC fit
    
    # Fourier power
    pwr = (np.absolute(np.fft.fft(data))) ** 2
    
    # Find the Fourier frequencies
    n = len(data)
    fftfreq = np.fft.fftfreq(n, dt)
    these_frequencies = fftfreq > 0
    
    # get the data at all the non-zero Fourier frequencies
    obs_freq = fftfreq[these_frequencies[:-1]]
    obs_pwr = pwr[these_frequencies[:-1]]

    # Set up the MCMC model
    M1 = pymc.MCMC(PyMCmodel(observed_frequencies=obs_freq,
                             observed_power=obs_pwr))

    # Run the sampler
    M1.sample(**kwargs)
    
    return M1
