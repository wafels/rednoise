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


def broken_power_law(f, a):
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
        a[3] : the break frequency.
    """
    # Normalize the frequencies
    fn = fnorm(f, f[0])
    # Calculate the break frequency
    fbreak = a[3]
    # Calculate the power normalization constant
    A = np.exp(a[0])
    # Create the output power array
    out = np.zeros(shape=f.shape)
    # Define the output power
    out[f < fbreak] = A * (fn[f < fbreak] ** (-a[1]))
    out[f >= fbreak] = A * (fn[f >= fbreak] ** (-a[2])) * (fbreak ** (a[2] - a[1]))
    return out


def splwc_GaussianBump(f, a):
    """Simple power law with a constant, and a Gaussian shaped bump.  This model assumes that the power
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
    return power_law_with_constant(f, a[0:3]) * np.exp(GaussianBump(np.log(f), a[3:6]))


def GaussianBump(x, a):
    z = (x - a[1]) / a[2]
    return a[0] * a[0] * np.exp(-0.5 * z ** 2)


def Log_splwc_GaussianBump(f, a):
    return np.log(power_law_with_constant(f, a[0:3])) + GaussianBump(np.log(f), a[3:6])


def Log_splwc(f, a):
    return np.log(power_law_with_constant(f, a))


def Log_splwc_GaussianBump_CF(f, a0, a1, a2, a3, a4, a5):
    return np.log(power_law_with_constant(f, [a0, a1, a2])) + GaussianBump(np.log(f), [a3, a4, a5])


def Log_splwc_CF(f, a0, a1, a2):
    return np.log(power_law_with_constant(f, [a0, a1, a2]))
