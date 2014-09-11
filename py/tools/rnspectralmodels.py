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


# ----------------------------------------------------------------------------
# Power law with constant
#
def power_law_with_constant(f, a):
    """Power law with a constant.  This model assumes that the power
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


# ----------------------------------------------------------------------------
# Normal distribution.
#
def NormalBump2(x, a):
    z = (x - a[1]) / a[2]
    amplitude = np.exp(a[0])
    norm = 1.0 / (np.sqrt(2 * np.pi * a[2] ** 2))
    return amplitude * norm * np.exp(-0.5 * z ** 2)


def NormalBump2_CF(x, a0, a1, a2):
    return NormalBump2(x, [a0, a1, a2])


def Log_NormalBump2_CF(x, a0, a1, a2):
    return np.log(NormalBump2(x, [a0, a1, a2]))


# ----------------------------------------------------------------------------
# Normal distribution, all exponential parameters.
#
def NormalBump2_allexp(x, a):
    amplitude = np.exp(a[0])
    position = np.exp(a[1])
    width = np.exp(a[2])
    z = (x - position) / width
    norm = 1.0 / (np.sqrt(2 * np.pi * width ** 2))
    return amplitude * norm * np.exp(-0.5 * z ** 2)


def NormalBump2_allexp_CF(x, a0, a1, a2):
    return NormalBump2_allexp(x, [a0, a1, a2])


def Log_NormalBump2_allexp_CF(x, a0, a1, a2):
    return np.log(NormalBump2_allexp(x, [a0, a1, a2]))


# ----------------------------------------------------------------------------
# Power law plus constant + lognormal bump.
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
def splwc_AddLognormalBump2(f, a):
    """Simple power law with a constant, plus a lognormally shaped bump.
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
        a[3] : the natural logarithm of the lognormal amplitude
        a[4] : the natural logarithm of the frequency of the center of the distribution
        a[5] : the width of the distribution in units of the natural logarithm of the frequency
    """
    return power_law_with_constant(f, a[0:3]) + NormalBump2(np.log(f), a[3:6])


def Log_splwc_AddLognormalBump2(f, a):
    return np.log(splwc_AddLognormalBump2(f, a))


def Log_splwc_AddLognormalBump2_CF(f, a0, a1, a2, a3, a4, a5):
    return Log_splwc_AddLognormalBump2(f, [a0, a1, a2, a3, a4, a5])


#
# ----------------------------------------------------------------------------
# This section implements the "non-periodic" component as described by
# Harvey 1993.
#
def exp_decay_autocor(f, a):
    return np.exp(a[0]) / (1.0 + (2 * np.pi * f / a[1]) ** a[2])


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


# ----------------------------------------------------------------------------
# Double broken power law model
#
def double_broken_power_law_with_constant(f, a):
    """
    a[0] = natural logarithm of the amplitude
    a[1] = power law index at frequencies less than a[3]
    a[2] = natural logarithm of the background constant
    a[3] = natural logarithm of the location of the break in the power law
    a[4] = power law index at frequencies greater than a[3]
    """
    power = np.float64(np.zeros_like(f))
    fbreak = np.exp(a[3])
    # Where the first power law is valid
    p1_location = f < fbreak
    # Where the second power law is valid
    p2_location = f >= fbreak
    # First power law
    p1 = np.exp(a[0]) * ((fnorm(f[p1_location], f[0]) ** (-a[1]))) + np.exp(a[2])
    # Second power law
    p2_amplitude = np.exp(a[0]) * fbreak ** (a[4] - a[1])
    p2 = p2_amplitude * ((fnorm(f[p2_location], f[0]) ** (-a[4]))) + np.exp(a[2])
    #Fill in the power
    power[p1_location] = p1
    power[p2_location] = p2
    return power


def Log_double_broken_power_law_with_constant(f, a):
    return np.log(double_broken_power_law_with_constant(f, a))


def Log_double_broken_power_law_with_constant_CF(f, a0, a1, a2, a3, a4):
    return np.log(double_broken_power_law_with_constant(f, [a0, a1, a2, a3, a4]))


# ----------------------------------------------------------------------------
# Power law plus Normal Bump
#
def splwc_AddNormalBump2(f, a):
    """Power law with a constant, plus a Gaussian shaped bump.

    Parameters
    ----------
    f : ndarray
        frequencies

    a : ndarray[2]
        a[0] : the natural logarithm of the normalization constant
        a[1] : the power law index
        a[2] : the natural logarithm of the constant background
        a[3] : the natural logarithm of the Gaussian amplitude
        a[4] : center of the Gaussian
        a[5] : width of the Gaussian
    """
    return power_law_with_constant(f, a[0:3]) + NormalBump2(f, a[3:6])


def Log_splwc_AddNormalBump2(f, a):
    return np.log(splwc_AddNormalBump2(f, a))


def Log_splwc_AddNormalBump2_CF(f, a0, a1, a2, a3, a4, a5):
    return Log_splwc_AddNormalBump2(f, [a0, a1, a2, a3, a4, a5])


# ----------------------------------------------------------------------------
# Power law plus Normal Bump
#
def splwc_AddNormalBump2_allexp(f, a):
    """Power law with a constant, plus a Gaussian shaped bump in which all the
    parameters have been expressed logarithmically.

    Parameters
    ----------
    f : ndarray
        frequencies

    a : ndarray[2]
        a[0] : the natural logarithm of the normalization constant
        a[1] : the power law index
        a[2] : the natural logarithm of the constant background
        a[3] : the natural logarithm of the Gaussian amplitude
        a[4] : center of the Gaussian
        a[5] : width of the Gaussian
    """
    return power_law_with_constant(f, a[0:3]) + NormalBump2_allexp(f, a[3:6])


def Log_splwc_AddNormalBump2_allexp(f, a):
    return np.log(splwc_AddNormalBump2_allexp(f, a))


def Log_splwc_AddNormalBump2_allexp_CF(f, a0, a1, a2, a3, a4, a5):
    return Log_splwc_AddNormalBump2_allexp(f, [a0, a1, a2, a3, a4, a5])
