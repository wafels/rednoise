"""
Power Spectrum Models
"""

import numpy as np


#
# Normalize the frequency
#
def fnorm(f, normalization):
    """ Normalize the frequency spectrum."""
    return f / normalization


# ----------------------------------------------------------------------------
# constant
#
def constant(a):
    """The power spectrum is a constant across all frequencies

    Parameters
    ----------
    a : float
        the natural logarithm of the power
    """
    return np.exp(a)


# ----------------------------------------------------------------------------
# Power law
#
def power_law(a, f):
    """Simple power law.  This model assumes that the power
    spectrum is made up of a power law at all frequencies.

    Parameters
    ----------
    a : ndarray[2]
        a[0] : the natural logarithm of the normalization constant
        a[1] : the power law index
    f : ndarray
        frequencies
    """
    return np.exp(a[0]) * (f ** (-a[1]))


# ----------------------------------------------------------------------------
# Power law with constant
#
def power_law_with_constant(a, f):
    """Power law with a constant.  This model assumes that the power
    spectrum is made up of a power law and a constant background.  At high
    frequencies the power spectrum is dominated by the constant background.

    Parameters
    ----------
    a : ndarray[2]
        a[0] : the natural logarithm of the normalization constant
        a[1] : the power law index
        a[2] : the natural logarithm of the constant background
    f : ndarray
        frequencies

    """
    return power_law(a[0:2], f) + constant(a[2])


# ----------------------------------------------------------------------------
# Lognormal
#
def lognormal(a, f):
    """
    A lognormal distribution

    Parameters
    ----------
    a : ndarray(3)
        a[0] : the natural logarithm of the Gaussian amplitude
        a[1] : the natural logarithm of the center of the Gaussian
        a[2] : the width of the Gaussian in units of natural logarithm of the
               frequency
    f : ndarray
        frequencies

    """
    onent = (np.log(f) - a[1]) / a[2]
    amp = np.exp(a[0])
    return amp * np.exp(-0.5 * onent ** 2)


# ----------------------------------------------------------------------------
# Delta function
#
def deltafn(a, f):
    """
    A delta function

    Parameters
    ----------
    a : ndarray(3)
        a[0] : the natural logarithm of the amplitude
        a[1] : the natural logarithm of the position of the delta function

    f : ndarray
        frequencies

    """
    delta_function = np.zeros(len(f))
    delta_function[np.argmin(np.abs(np.log(f) - a[1]))] = 1.0
    return np.exp(a[0]) * delta_function


# ----------------------------------------------------------------------------
# Power law with lognormal
#
def power_law_with_constant_with_lognormal(a, f):
    """
    Power law with constant and a lognormal.

    Parameters
    ----------
    a : ndarray[6]
        a[0] : the natural logarithm of the normalization constant
        a[1] : the power law index
        a[2] : the natural logarithm of the constant background
        a[3] : the natural logarithm of the Gaussian amplitude
        a[4] : the natural logarithm of the center of the Gaussian
        a[5] : the width of the Gaussian in units of natural logarithm of the
               frequency
    f : ndarray
        frequencies
    """
    return power_law_with_constant(a[0:3], f) + lognormal(a[3:6], f)


# ----------------------------------------------------------------------------
# Power law with delta function
#
def power_law_with_constant_with_deltafn(a, f):
    """
    Power law with constant and a delta function.

    Parameters
    ----------
    a : ndarray[5]
        a[0] : the natural logarithm of the normalization constant
        a[1] : the power law index
        a[2] : the natural logarithm of the constant background
        a[3] : the natural logarithm of the delta function amplitude
        a[4] : the natural logarithm of the position of the delta function

    f : ndarray
        frequencies
    """
    return power_law_with_constant(a[0:3], f) + deltafn(a[3:4], f)
