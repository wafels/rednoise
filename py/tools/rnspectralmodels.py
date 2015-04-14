"""
Power Spectrum Models
"""

import numpy as np


#
# Normalize the frequency
#
def fnorm(f, fnorm):
    """ Normalize the frequency spectrum."""
    return f / fnorm


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
    return np.exp(a[0]) * (fnorm(f, f[0]) ** (-a[1]))


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
    return power_law(a[0:2], f) + np.exp(a[2])


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
    fn = fnorm(f, f[0])
    onent = (np.log(fn) - a[4]) / a[5]
    amp = np.exp(a[3])
    lognormal_term = amp * np.exp(-0.5 * onent ** 2)
    return power_law_with_constant(a[0:3], f) + lognormal_term


# ----------------------------------------------------------------------------
# Power law with lognormal
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
    fn = fnorm(f, f[0])
    delta_function = np.zeros(len(fn))
    delta_function[np.argmin(np.abs(np.log(fn) - a[4]))] = 1.0
    return power_law_with_constant(a[0:3], f) + np.exp(a[3]) * delta_function
