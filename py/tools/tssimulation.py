"""
Test the power spectrum tools
"""

# Test 1
import rnspectralmodels
import rnsimulation
import numpy as np


def time_series(nt, dt, A_osc, B_osc, frequency, polynomial, V=100, W=100):
    """
    Defines a simple time series with a background trend and a frequency
    """
    # Define the white noise power spectrum
    P = rnsimulation.ConstantSpectrum(parameters, nt=nt, dt=dt)

    # Get a time series that is noisy
    white_noise = rnsimulation.TimeSeriesFromPowerSpectrum(P, V=W, W=W).sample

    # Sample times
    t = dt * np.arange(0, nt)

    return oscillation(A_osc, B_osc, frequency, t) + \
        trend(polynomial, t) + \
        white_noise


def oscillation(A, B, frequency, t):
    """
    Define a single sinusoidal oscillation
    """
    return A * np.sin(2 * np.pi * frequency * t) + \
        B * np.cos(2 * np.pi * frequency * t)


def trend(polynomial, t):
    """
    Define a polynomial background trend
    """
    return np.polyval(polynomial, t)
