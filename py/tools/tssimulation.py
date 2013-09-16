"""
Simulation of time series
"""

import numpy as np


def sinusoid(A, B, frequency, t):
    """
    Define a single sinusoidal oscillation of the form
    A sin( 2 pi f t ) + B sin( 2 pi f t )
    """
    return A * np.sin(2 * np.pi * frequency * t) + \
        B * np.cos(2 * np.pi * frequency * t)


def trend(polynomial, t):
    """
    Define a polynomial background trend
    """
    return np.polyval(polynomial, t)
