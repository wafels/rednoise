"""
Simulation of time series
"""

import rnsimulation
import numpy as np


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
