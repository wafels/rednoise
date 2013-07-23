"""
Spectra for use with the red noise MCMC analysis
"""

import numpy as np


def fnorm(f, fnorm):
    """ Normalize the frequency spectrum"""
    return f / fnorm


def constant(f, a):
    """Constant power at all frequencies

    Parameters
    ----------
    f : ndarray
        frequencies

    a : scalar number
        the constant value of the power
    """
    return a * np.ones(f.shape)


def power_law(f, a):
    """Simple power law.  This model assumes that the power
    spectrum is made up of a power law at all frequencies.

    Parameters
    ----------
    f : ndarray
        frequencies

    a : ndarray[2]
        a[0] : the natural logarithm of the normalization constant
        a[1] : the power law index
    """
    return np.exp(a[0]) * ((fnorm(f, f[0]) ** (-a[1])))


def power_law_with_constant(f, a):
    """Simple power law with a constant.  This model assumes that the power
    spectrum is made up of a power law and a constant background.  At high
    frequencies the power spectrum is dominated by the constant background.

    Parameters
    ----------
    f : ndarray
        frequencies

    a : ndarray[2]
        a[0] : the natural logarithm of the normalization constant
        a[1] : the power law index
        a[2] : the natural logarithm of the constant background
    """
    return power_law(f, a[0:2]) + np.exp(a[2])


def broken_power_law_log_break_frequency(f, a):
    """Broken power law.  This model assumes that the power
    spectrum is made up of a power law with a given index below a frequency f0,
    and a power law with another index above the frequency f0.

    Parameters
    ----------
    f : ndarray
        frequencies

    a : ndarray[2]
        a[0] : the natural logarithm of the normalization constant
        a[1] : the power law index below the value of f0
        a[2] : the power law index below the value of f0
        a[3] : the natural logarithm of the break frequency.
    """
    # Normalize the frequencies
    fn = fnorm(f, f[0])
    # Calculate the break frequency
    fbreak = np.exp(a[3])
    # Calculate the power normalization constant
    A = np.exp(a[0])
    # Create the output power array
    out = np.zeros(shape=f.shape)
    # Define the output power
    out[f < fbreak] = A * (fn[f < fbreak] ** (-a[1]))
    out[f >= fbreak] = A * (fn[f >= fbreak] ** (-a[2])) * (fbreak ** (a[2] - a[1]))
    return out