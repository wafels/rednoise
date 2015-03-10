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
# Power law with Gaussian amplitude
#
def power_law_with_gaussian(a, f):
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
    fn = fnorm(f, f[0])
    plaw = power_law(a[0:3], f)
    onent = (np.log(fn) - a[3]) / a[4]
    amp = np.exp(a[2])
    gterm = amp * np.exp(-0.5 * onent ** 2)
    return plaw + gterm


#