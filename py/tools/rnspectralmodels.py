"""
Spectra for use with the red noise MCMC analysis
"""

import numpy as np


#
# Model without bump
#
def fnorm(f, fnorm):
    """ Normalize the frequency spectrum"""
    return f / fnorm


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


def Log_splwc(f, a):
    return np.log(power_law_with_constant(f, a))


def Log_splwc_CF(f, a0, a1, a2):
    return np.log(power_law_with_constant(f, [a0, a1, a2]))


#
# Model with normal bump.
#
# This model adds a lognormally distributed bump
# to the data. Note that this particular implementation uses the full
# definition of a normal distribution.  This means that the term multiplying
# the exponential depends both on an amplitude variable and a width variable.
# The advantage of doing this is that the fit value to the amplitude is equal
# integral under the curve of the lognormal distribution.  This is a way of
# getting the total power due to the Gaussian bump.  A potential disadvantage
# of this approach is that it introduces extra dependencies in calculating the
# amplitude term.  This means the search space is slightly more coupled than
# the implementation below "Model with Gaussian Bump"
#
def NormalBump2(x, a):
    z = (x - a[1]) / a[2]
    amplitude = np.exp(a[0])
    norm = 1.0 / (np.sqrt(2 * np.pi * a[2] ** 2))
    return amplitude * norm * np.exp(-0.5 * z ** 2)


def NormalBump2_CF(x, a0, a1, a2):
    z = (x - a1) / a2
    amplitude = np.exp(a0)
    norm = 1.0 / (np.sqrt(2 * np.pi * a2 ** 2))
    return amplitude * norm * np.exp(-0.5 * z ** 2)


def splwc_AddNormalBump2(f, a):
    """Simple power law with a constant, plus a Gaussian shaped bump.
    This model assumes that the powe spectrum is made up of a power law and a
    constant background.  At high frequencies the power spectrum is dominated
    by the constant background.

    Parameters
    ----------
    f : ndarray
        frequencies

    a : ndarray[2]
        a[0] : the natural logarithm of the normalization constant
        a[1] : the power law index
        a[2] : the natural logarithm of the constant background
    """
    return power_law_with_constant(f, a[0:3]) + NormalBump2(np.log(f), a[3:6])


def Log_splwc_AddNormalBump2(f, a):
    """Simple power law with a constant, plus a Gaussian shaped bump.
    This model assumes that the powe spectrum is made up of a power law and a
    constant background.  At high frequencies the power spectrum is dominated
    by the constant background.

    Parameters
    ----------
    f : ndarray
        frequencies

    a : ndarray[2]
        a[0] : the natural logarithm of the normalization constant
        a[1] : the power law index
        a[2] : the natural logarithm of the constant background
    """
    return np.log(splwc_AddNormalBump2(f, a))


def Log_splwc_AddNormalBump2_CF(f, a0, a1, a2, a3, a4, a5):
    return Log_splwc_AddNormalBump2(f, [a0, a1, a2, a3, a4, a5])


#
# This section implements the "non-periodic" component as described by
# Harvey 1993.
#
def exp_decay_autocor(f, a):
    return np.exp(a[0]) / (1.0 + (2 * np.pi * f * a[1]) ** a[2])


def exp_decay_autocor_CF(f, a0, a1, a2):
    return exp_decay_autocor(f, [a0, a1, a2])


def splwc_AddExpDecayAutocor(f, a):
    """Simple power law with a constant, a model component that is constant at
    low frequencies and tails off to zero at high frequencies.  At high
    frequencies the power spectrum is dominated by the constant background.

    Parameters
    ----------
    f : ndarray
        frequencies

    a : ndarray[6]
        a[0] : the natural logarithm of the normalization constant
        a[1] : the power law index
        a[2] : the natural logarithm of the constant background
        a[3] : the natural logarithm of the normalization constant for the
               second component
        a[4] : period
        a[5] : power law of decay
    """
    return power_law_with_constant(f, a[0:3]) + exp_decay_autocor(f, a[3:6])


def Log_splwc_AddExpDecayAutocor(f, a):
    return np.log(splwc_AddExpDecayAutocor(f, a))


def Log_splwc_AddExpDecayAutocor_CF(f, a0, a1, a2, a3, a4, a5):
    return Log_splwc_AddExpDecayAutocor(f, [a0, a1, a2, a3, a4, a5])

#
#
#
# ----------------------------------------------------------------------------
# Everything below here can probably be safely deleted.
#
#
# Model with Gaussian bump.
#
#
def GaussianBump2(x, a):
    z = (x - a[1]) / a[2]
    amplitude = np.exp(a[0])
    return amplitude * np.exp(-0.5 * z ** 2)


def GaussianBump2_CF(x, a0, a1, a2):
    z = (x - a1) / a2
    amplitude = np.exp(a0)
    return amplitude * np.exp(-0.5 * z ** 2)


def splwc_AddGaussianBump2(f, a):
    """Simple power law with a constant, plus a Gaussian shaped bump.
    This model assumes that the powe spectrum is made up of a power law and a
    constant background.  At high frequencies the power spectrum is dominated
    by the constant background.

    Parameters
    ----------
    f : ndarray
        frequencies

    a : ndarray[2]
        a[0] : the natural logarithm of the normalization constant
        a[1] : the power law index
        a[2] : the natural logarithm of the constant background
    """
    return power_law_with_constant(f, a[0:3]) + GaussianBump2(np.log(f), a[3:6])


def Log_splwc_AddGaussianBump2(f, a):
    """Simple power law with a constant, plus a Gaussian shaped bump.
    This model assumes that the powe spectrum is made up of a power law and a
    constant background.  At high frequencies the power spectrum is dominated
    by the constant background.

    Parameters
    ----------
    f : ndarray
        frequencies

    a : ndarray[2]
        a[0] : the natural logarithm of the normalization constant
        a[1] : the power law index
        a[2] : the natural logarithm of the constant background
    """
    return np.log(splwc_AddGaussianBump2(f, a))


def Log_splwc_AddGaussianBump2_CF(f, a0, a1, a2, a3, a4, a5):
    return np.log(power_law_with_constant(f, [a0, a1, a2]) + NormalBump2(np.log(f), [a3, a4, a5]))
