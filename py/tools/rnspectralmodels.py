"""
Power Spectrum Models
"""

import numpy as np
import lnlike_model_fit
import pstools


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


def lognormal_CF(f, a, b, c):
    return lognormal([a, b, c], f)


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


class CompoundSpectrum:
    def __init__(self, components):
        self.components = components
        self.names = []
        self.labels = []
        for component in self.components:
            spectrum = component[0]
            for name in spectrum.names:
                self.names.append(name)
            for label in spectrum.labels:
                self.labels.append(label)

    def power(self, a, f):
        total_power = np.zeros_like(f)
        for component in self.components:
            spectrum = component[0]
            variables = a[component[1][0]: component[1][1]]
            total_power += spectrum.power(variables, f)
        return total_power

    def guess(self, f, data):
        pass


class Constant:
    def __init__(self):
        self.names = ['log(constant)']
        self.labels = ['$\log(constant)$']

    def power(self, a, f):
        return constant(a)


class PowerLaw:
    def __init__(self):
        self.names = ['log(power law amplitude)', 'power law index']
        self.labels = [r'$\log(A)$', r'$n$']

    def power(self, a, f):
        return power_law(a, f)


class Lognormal:
    def __init__(self):
        self.names = ['log(lognormal amplitude)', 'lognormal location', 'lognormal width']
        self.labels = ['$\log(A_{L})$', '$p_{L}$', '$w_{L}$']

    def power(self, a, f):
        return lognormal(a, f)


class PowerLawPlusConstant(CompoundSpectrum):
    def __init__(self):
        CompoundSpectrum.__init__(self, ((PowerLaw(), (0, 2)),
                                         (Constant(), (2, 3))))

    def guess(self, f, power, amp_range=[0, 5], index_range=[0, 50],
              background_range=[-50, -1]):

        log_amplitude_estimate = np.log(np.mean(power[amp_range[0]:amp_range[1]]))
        index_estimate = pstools.most_probable_power_law_index(f[index_range[0]:index_range[1]],
                                                       power[index_range[0]:index_range[1]],
                                                       0.0, np.arange(0.0, 4.0, 0.01))
        log_background_estimate = np.log(np.mean(power[background_range[0]:background_range[1]]))
        return log_amplitude_estimate, index_estimate, log_background_estimate


class PowerLawPlusConstantPlusLognormal(CompoundSpectrum):
    def __init__(self):
        CompoundSpectrum.__init__(self, ((PowerLaw(), (0, 2)),
                                         (Constant(), (2, 3)),
                                         (Lognormal(), (3, 6))))

    def guess(self, f, power, amp_range=[0, 5], index_range=[0, 50],
              background_range=[-50, -1], f_lower_limit=50.0,
              f_upper_limit=200.0, sufficient_frequencies=10,
              initial_log_width=0.1):

        log_amplitude, index_estimate, log_background = PowerLawPlusConstant.guess().guess(f, power)
        background_spectrum = PowerLawPlusConstant.power([log_amplitude, index_estimate, log_background], f)

        # Difference between the data and the model
        diff0 = power - background_spectrum

        # Keep the positive parts only
        positive_index = diff0 > 0.0

        # Limit the fit to a specific frequency range
        f_above = f > f_lower_limit
        f_below = f < f_upper_limit

        # Which data to fit
        fit_here = positive_index * f_above * f_below

        # If there is sufficient positive data
        if len(fit_here) > sufficient_frequencies:
            diff1 = diff0[fit_here]
            f1 = f[fit_here]
            amp = np.log(np.max(diff1))
            pos = np.log(f1[np.argmax(diff1)])
            initial_guess = [log_amplitude, index_estimate, log_background, amp, pos, initial_log_width]
        else:
            initial_guess = [log_amplitude, index_estimate, log_background,
                             -100.0,
                             0.5 * (f_lower_limit + f_upper_limit),
                             initial_log_width]
        return initial_guess


#
# A class that fits models to data and calculates various fit measures.
#
class Fit:
    def __init__(self, f, data, model, guess=None, fit_method='Nelder-Mead',
                 **kwargs):

        # Frequencies
        self.f = f

        # Number of data points
        self.n = len(self.f)

        # Model that we are going to fit to the data
        self.model = model

        # Fit Method
        self.fit_method = fit_method

        # Initial guess
        if guess is None:
            self.guess = model.guess(self.f, data, **kwargs)
        else:
            self.guess = guess

        # Number of free parameters
        self.k = len(self.guess)

        # Degrees of freedom
        self.dof = self.n - self.k - 1

        # Get the fit results
        self.result = lnlike_model_fit.go(self.f, data, self.power,
                                          self.guess, self.fit_method)

        self.estimate = self.result['x']

        self.best_fit = self.model.power(self.estimate, self.f)

        self.rhoj = lnlike_model_fit.rhoj(data, self.best_fit)

        self.rchi2 = lnlike_model_fit.rchi2(1, self.dof, self.rhoj)

        self.AIC = lnlike_model_fit.AIC(self.k, self.estimate, self.f, data, self.model)

        self.BIC = lnlike_model_fit.BIC(self.k, self.estimate, self.f, data, self.model, self.n)
