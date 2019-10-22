"""
Power Spectrum Models
"""
from copy import deepcopy
import datetime
import numpy as np
from scipy.optimize import curve_fit
import astropy.units as u
from tools import lnlike_model_fit
from tools import pstools

import matplotlib.pyplot as plt
#
# Magic numbers.  These frequency limits correspond to
# 0.1 mHz to 10 mHz when we use 6 hours of AIA data, corresponding
# to 1800 frames at 12 second cadence.
#
magic_number_lognormal_position_frequency_lower_limit = 1 * u.Hz / 1000.0
magic_number_lognormal_position_frequency_upper_limit = 10.0 * u.Hz / 1000.0

#
# These magic numbers cover the width of the lognormal.
#
magic_number_lognormal_width_lower_limit = 0.01
magic_number_lognormal_width_upper_limit = 0.5


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
# Broken Power law
#
def broken_power_law(a, f):
    """Broken power law.  This model assumes that there is a break in the power
    spectrum at some given frequency.

    Parameters
    ----------
    a : ndarray[3]
        a[0] : the natural logarithm of the normalization constant
        a[1] : the power law index at frequencies lower than the break frequency
        a[2] : break frequency
        a[3] : the power law index at frequencies higher than the break frequency
    f : ndarray
        frequencies
    """
    power = np.zeros_like(f)
    less_than_break = f < a[2]
    above_break = f >= a[2]
    power[less_than_break] = np.exp(a[0]) * f[less_than_break] ** (-a[1])
    power[above_break] = np.exp(a[0]) * (a[2]**(-a[1]+a[3])) * f[above_break] ** (-a[3])
    return power


# ----------------------------------------------------------------------------
# Sum of Pulses
#
def sum_of_pulses(a, f):
    """Sum of pulses.  This model is based on Aschwanden "Self Organized
    Criticality in Astrophysics", Eq. 4.8.23.  Simulations implementing this
    equation come up with a shape that is modeled below.

    Parameters
    ----------
    a : ndarray[3]
        a[0] : the natural logarithm of the normalization constant
        a[1] : the scale frequency
        a[2] : the power law index
    f : ndarray
        frequencies
    """
    return np.exp(a[0])/(1.0 + (f/a[1]) ** a[2])


# ----------------------------------------------------------------------------
# Sum of Pulses Plus Constant
#
def sum_of_pulses_with_constant(a, f):
    """Sum of pulses plus constant.  This model is based on Aschwanden "Self
    Organized Criticality in Astrophysics", Eq. 4.8.23, with a constant
    background to model detector noise.

    Parameters
    ----------
    a : ndarray[3]
        a[0] : the natural logarithm of the normalization constant
        a[1] : the scale frequency
        a[2] : the power law index
        a[3] : natural logarithm of the background constant
    f : ndarray
        frequencies
    """
    return np.exp(a[0])/(1.0 + (f/a[1]) ** a[2]) + np.exp(a[3])


# ----------------------------------------------------------------------------
# Broken Power law with Constant
#
def broken_power_law_with_constant(a, f):
    """Broken power law with constant.  This model assumes that there is a
    break in the power spectrum at some given frequency.  At high
    frequencies the power spectrum is dominated by the constant background.

    Parameters
    ----------
    a : ndarray(5)
        a[0] : the natural logarithm of the normalization constant
        a[1] : the power law index at frequencies lower than the break frequency
        a[2] : break frequency
        a[3] : the power law index at frequencies higher than the break frequency
        a[4] : the natural logarithm of the constant background
    f : ndarray
        frequencies
    """
    return broken_power_law(a[0:4], f) + constant(a[4])


# ----------------------------------------------------------------------------
# Broken Power law with Constant
#
def broken_power_law_with_constant_with_lognormal(a, f):
    """Broken power law with constant with lognormal.  This model assumes that
    there is a break in the power spectrum at some given frequency.  At high
    frequencies the power spectrum is dominated by the constant background.  At
    some particular frequency there is a lognormal (narrowband distribution)

    Parameters
    ----------
    a : ndarray(5)
        a[0] : the natural logarithm of the normalization constant
        a[1] : the power law index at frequencies lower than the break frequency
        a[2] : break frequency
        a[3] : the power law index at frequencies higher than the break frequency
        a[4] : the natural logarithm of the constant background
    f : ndarray
        frequencies
    """
    return broken_power_law(a[0:4], f) + constant(a[4])


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


#
# Spectrum model class
#
class Spectrum:
    def __init__(self, name, variables, f_norm=1*u.Hz):
        # Spectral model name
        self.name = name

        # Variables of the spectral model.  The variables listed here are
        # instances or subclasses of the Variable class.
        self.variables = variables

        # Normalization applied to the frequencies
        if not isinstance(f_norm, u.Quantity):
            raise ValueError('Normalization factor must be a quantity and convertible to Hz.')
        self.f_norm = f_norm

    def power(self, a, f):
        if isinstance(f, u.Quantity):
            raise ValueError('Spectrum power argument must be a normalized frequency.')
        pass

    def guess(self, f, data):
        if isinstance(f, u.Quantity):
            raise ValueError('Spectrum power argument must be a normalized frequency.')
        pass


class BrokenPowerLaw(Spectrum):
    def __init__(self):

        power_law_amplitude = LnVariable('power law amplitude', 'A_{P}')
        power_law_index_below_break = Variable('power law index below break',
                                               no_conversion,
                                               r'$n_{below}$',
                                               u.dimensionless_unscaled)
        break_frequency = FrequencyVariable(r"$\nu_{break}$")
        power_law_index_above_break = Variable('power law index above break',
                                               no_conversion,
                                               r'$n_{above}$',
                                               u.dimensionless_unscaled)

        Spectrum.__init__(self, 'broken power law',
                          [power_law_amplitude, power_law_index_below_break,
                           break_frequency, power_law_index_above_break])

    def power(self, a, f):
        return broken_power_law(a, f)

    # A good enough guess that will allow a proper fitting algorithm to proceed.
    def guess(self, f, observed_power, amp_range=[0, 5],
              break_frequency=100):
        log_amplitude_estimate, index_estimate_below = PowerLaw.guess(f[0:break_frequency],
                                                                      observed_power[0:break_frequency],
                                                                      amp_range=amp_range)
        _, index_estimate_above = PowerLaw.guess(f[break_frequency:],
                                                 observed_power[break_frequency:],
                                                 amp_range=amp_range)

        return log_amplitude_estimate, index_estimate_below, break_frequency, index_estimate_above


class Lognormal(Spectrum):
    def __init__(self):

        amplitude = LnVariable('lognormal amplitude', 'A_{L}')
        position = Variable('lognormal position', convert_ln_dimensionless_frequency_to_frequency, r"$p_{L}$", u.Hz)
        width = LnVariable('lognormal width', 'w_{L}')

        Spectrum.__init__(self, 'Lognormal', [amplitude, position, width])

    def power(self, a, f):
        return lognormal(a, f)


class PowerLawPlusConstant(CompoundSpectrum):
    def __init__(self, f_norm=1.0*u.Hz):
        CompoundSpectrum.__init__(self, ((PowerLaw(), (0, 2)),
                                         (Constant(), (2, 3))))
        self.f_norm = f_norm

    def guess(self, f, power, amp_range=[0, 5], index_range=[0, 50],
              background_range=[-50, -1]):

        log_amplitude_estimate = np.log(np.mean(power[amp_range[0]:amp_range[1]]))
        index_estimate = pstools.most_probable_power_law_index(f[index_range[0]:index_range[1]],
                                                       power[index_range[0]:index_range[1]],
                                                       0.0, np.arange(0.0, 4.0, 0.01))
        log_background_estimate = np.mean(np.log(power[background_range[0]:background_range[1]]))
        return log_amplitude_estimate, index_estimate, log_background_estimate

    def acceptable_fit(self, a):
        return True

    def vary_guess(self, a):
        return a


class PowerLawPlusConstantPlusLognormal(CompoundSpectrum):
    def __init__(self, f_norm=1.0*u.Hz):
        CompoundSpectrum.__init__(self, ((PowerLaw(), (0, 2)),
                                         (Constant(), (2, 3)),
                                         (Lognormal(), (3, 6))))
        self.f_norm = f_norm
        self.amp_range = [0, 5]
        self.index_range = [0, 50]
        self.background_range = [-50, -1]
        self.f_lower_limit = (magic_number_lognormal_position_frequency_lower_limit / self.f_norm).value
        self.f_upper_limit = (magic_number_lognormal_position_frequency_upper_limit / self.f_norm).value
        self.width_lower_limit = magic_number_lognormal_width_lower_limit
        self.width_upper_limit = magic_number_lognormal_width_upper_limit
        self.sufficient_frequencies = 10
        self.initial_log_width = 0.1

    def acceptable_fit(self, a):
        center_condition = (np.log(self.f_lower_limit) <= a[4]) and (np.log(self.f_upper_limit) >= a[4])
        width_condition = (self.width_lower_limit <= a[5]) and (self.width_upper_limit >= a[5])
        return center_condition and width_condition

    def vary_guess(self, a):
        new_a = deepcopy(a)
        vary_range = 0.1 * (np.log(self.f_upper_limit) - np.log(self.f_lower_limit))
        while not self.acceptable_fit(new_a):
            if a[4] <= np.log(self.f_lower_limit):
                new_a[4] = np.log(self.f_lower_limit) + vary_range*np.random.uniform()
            if a[4] >= np.log(self.f_upper_limit):
                new_a[4] = np.log(self.f_upper_limit) - vary_range*np.random.uniform()
        return new_a

    def guess(self, f, power):

        # Initial estimate for the background power law power spectrum
        log_amplitude, index_estimate, log_background = PowerLawPlusConstant().guess(f, power)

        # Should use the above guess to seed a fit for PowerLawPlusConstant
        # based on the excluded estimated location of the lognormal
        background_spectrum = PowerLawPlusConstant().power([log_amplitude, index_estimate, log_background], f)

        # Define a default guess if we can't fit a lognormal
        default_guess = [log_amplitude, index_estimate, log_background,
                         -100.0,
                         0.5 * (np.log(self.f_lower_limit) + np.log(self.f_upper_limit)),
                         self.initial_log_width]

        # Sanity check on the default guess
        if not self.acceptable_fit(default_guess):
            print(default_guess)
            raise ValueError('The default guess does not satisfy the acceptable fit criterion.')

        # Let's see if we can fit a lognormal
        # Difference between the data and the model
        diff0 = power - background_spectrum

        # Keep the positive parts only
        positive_index = diff0 > 0.0

        # Limit the fit to a specific frequency range
        f_above = f > self.f_lower_limit
        f_below = f < self.f_upper_limit

        # Which data to fit
        fit_here = positive_index * f_above * f_below

        # If there is sufficient positive data indicating a bump, try fitting a lognormal distribution.
        # If the fit is not acceptable, return the default guess.
        # If there is insufficient positive data, return the default guess.
        # If the fit fails
        if np.sum(fit_here) > self.sufficient_frequencies:
            positive_data = diff0[fit_here]
            frequencies_of_positive_data = f[fit_here]
            try:
                popt, pcov = curve_fit(lognormal_CF, frequencies_of_positive_data, positive_data)
            except RuntimeError:
                return default_guess
            except ValueError:
                return default_guess
            initial_guess = [log_amplitude, index_estimate, log_background, popt[0], popt[1], abs(popt[2])]
            if not self.acceptable_fit(initial_guess):
                return default_guess
            else:
                return initial_guess
        else:
            return default_guess

#
#
#
def estimate_power_law_plus_constant(f, power,
                                     amplitude_range=(0, 5),
                                     index_range=(0, 50),
                                     background_range=(-50, -1)):
    """

    Parameters
    ----------
    f
    power
    amplitude_range
    index_range
    background_range

    Returns
    -------

    """
    # Amplitude estimate
    amplitude_estimate = np.mean(power[amplitude_range[0]:amplitude_range[1]])

    # Index estimate
    index_estimate = pstools.most_probable_power_law_index(f[index_range[0]:index_range[1]],
                                                           power[index_range[0]:index_range[1]],
                                                           0.0, np.arange(0.0, 4.0, 0.01))
    # Background estimate
    background_estimate = np.mean(power[background_range[0]:background_range[1]])
    return amplitude_estimate, index_estimate, background_estimate


#
# A function that fits a power law plus a constant to
#
def fit_power_law_plus_constant(f, power, fit_method='Nelder-Mead', **kwargs):
        """

        Parameters
        ----------
        f
        power
        fit_method
        kwargs

        Returns
        -------

        """
        # Number of data points in each power spectrum
        n = len(f)

        # Number of free parameters
        k = 3

        # Degrees of freedom
        dof = n - k - 1

        # Where the interesting parameters are stored in the storage array
        self.index = {"success": [1, 'success'],
                      "rchi2": [2],
                      "AIC": [3],
                      "BIC": [4]}
        for i, variable in enumerate(model.variables):
            self.index[variable.fit_parameter] = [1, 'x', i]


        # Initial guess should always satisfy the acceptable fit criterion
        # as defined in the model.
        estimate = estimate_power_law_plus_constant(f, power, **kwargs)

        # We want to ensure that at least one fit is attempted using the
        # initial guess.  To do this, the acceptable fit property is set
        # to False.
        acceptable_fit_found = False
        this_guess = deepcopy(guess)

        # Vary the initial guess until a good fit is found up to a
        result = lnlike_model_fit.go(f, power,
                                     self.model.power,
                                     estimate,
                                     fit_method)

        # Estimates of the quality of the fit to the data
        parameter_estimate = result['x']

        bestfit = self.model.power(parameter_estimate, self.fn)
        rhoj = lnlike_model_fit.rhoj(observed_power, bestfit)
        rchi2 = lnlike_model_fit.rchi2(1.0, dof, rhoj)
        aic = lnlike_model_fit.AIC(k, parameter_estimate, self.fn,
                                   observed_power, self.model.power)
        bic = lnlike_model_fit.BIC(self.k, parameter_estimate, self.fn,
                                   observed_power, self.model.power,
                                   self.n)

        # Store the final results
        self.result[j][i] = (this_guess, result, rchi2, aic, bic, self.acceptable_fit_found)




    def good_rchi2_mask(self, p_value=[0.025, 0.975]):
        """
        Calculate a numpy mask value array that indicates which positions in
        the results array have good fits in the sense that they lie inside the
        a range of reduced chi-squared defined by the passed in p.values.
        Where mask = False indicates a GOOD fit, mask = True indicates a BAD
        fit.  This can be passed directly into numpy's masked array.

        :param p_value: Two element object that has the lower and upper p values
        which are used to calculate corresponding reduced chi-squared values.

        :return: A logical mask where mask = False indicates a GOOD fit,
        mask = True indicates a BAD fit.  This can be passed directly into
        numpy's masked array.
        """
        rchi2 = self.as_array("rchi2")
        rchi2_gt_low_limit = rchi2 > self._rchi2limit(p_value[1])
        rchi2_lt_high_limit = rchi2 < self._rchi2limit(p_value[0])
        return np.logical_not(rchi2_gt_low_limit * rchi2_lt_high_limit)

    def _rchi2limit(self, p_value):
        return lnlike_model_fit.rchi2_given_prob(p_value, 1.0, self.dof)

    def good_fits(self, p_value=[0.025, 0.975]):
        """
        Find out where the good fits are.  Good fits are defined as those that
        have a reduced-chi-squared within the range defined by the input
        p values, and a successful fit as defined by the fitting algorithm.

        :param p_value: Two element object that has the lower and upper p values
        which are used to calculate corresponding reduced chi-squared values.
        :return: A logical mask where mask = False indicates a GOOD fit,
        mask = True indicates a BAD fit.  This can be passed directly into
        numpy's masked array.
        """
        return self.good_rchi2_mask(p_value) * self.fitting_algorithm_success()

    def fitting_algorithm_success(self):
        return self.as_array("success")

    def best_fit(self):
        """
        Calculate the best fit power spectrum at each pixel
        :return: A three dimensional numpy array that has the best fit at each
        spatial location.
        """
        bf = np.zeros((self.ny, self.nx, self.n))
        for i in range(0, self.nx):
            for j in range(0, self.ny):
                bf[j, i, :] = self.model.power(self.result[j][i][1]['x'], self.fn)
        return bf
