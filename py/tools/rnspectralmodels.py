"""
Power Spectrum Models
"""

import numpy as np
import astropy.units as u
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
        self.name = ''
        self.parameters = []
        self.labels = []
        self.conversion = []
        for component in self.components:
            spectrum = component[0]
            self.name = self.name + spectrum.name + ' + '
            for name in spectrum.parameters:
                self.parameters.append(name)
            for label in spectrum.labels:
                self.labels.append(label)
            for conversion in spectrum.conversion:
                self.conversion.append(conversion)
        self.name = self.name[:-3]

    # Calculate the power in each component.  Useful for plotting out the
    # individual components, ratios, etc
    def power_per_component(self, a, f):
        ppc = []
        for component in self.components:
            spectrum = component[0]
            variables = a[component[1][0]: component[1][1]]
            ppc.append(spectrum.power(variables, f))
        return ppc

    # Return the total power
    def power(self, a, f):
        ppc = self.power_per_component(a, f)
        total_power = np.zeros_like(f)
        for power in range(0, len(self.components)):
            total_power += ppc[power]
        return total_power

    # Subclassed for any particular power law
    def guess(self, f, data):
        pass


class Constant:
    def __init__(self):
        self.name = 'Constant'
        self.parameters = ['log10(constant)']
        self.labels = ['$\log_{10}(C)$']
        self.conversion = [1.0/np.log(10.0)]

    def power(self, a, f):
        return constant(a)


class PowerLaw:
    def __init__(self):
        self.name = 'Power law'
        self.parameters = ['log10(power law amplitude)', 'power law index']
        self.labels = [r'$\log_{10}(A_{P})$', r'$n$']
        self.conversion = [1.0/np.log(10.0), 1.0]

    def power(self, a, f):
        return power_law(a, f)


class Lognormal:
    def __init__(self):
        self.name = 'Lognormal'
        self.parameters = ['log10(lognormal amplitude)', 'log10(lognormal position)', 'log10(lognormal width)']
        self.labels = ['$\log_{10}(A_{L})$', '$\log_{10}(p_{L})$', '$\log_{10}(w_{L})$']
        self.conversion = [1.0/np.log(10.0), 1.0/np.log(10.0), 1.0/np.log(10.0)]

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

        log_amplitude, index_estimate, log_background = PowerLawPlusConstant().guess(f, power)
        #
        # Should use the above guess to seed a fit for PowerLawPlusConstant
        # based on the excluded estimated location of the lognormal
        #
        background_spectrum = PowerLawPlusConstant().power([log_amplitude, index_estimate, log_background], f)

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
        if np.sum(fit_here) > sufficient_frequencies:
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
    def __init__(self, f, data, model, fit_method='Nelder-Mead',
                 **kwargs):

        # Frequencies
        self.f = f

        # Normalized frequencies
        if isinstance(self.f, u.Quantity):
            self.fn = (self.f / self.f[0]).value
        else:
            self.fn = self.f / self.f[0]

        # Number of data points in each power spectrum
        self.n = len(self.f)

        # Model that we are going to fit to the data
        self.model = model

        # Fit Method
        self.fit_method = fit_method

        # Number of free parameters
        self.k = len(self.model.parameters)

        # Degrees of freedom
        self.dof = self.n - self.k - 1

        # Where the interesting parameters are stored in the storage array
        self.index = {"success": [1, 'success'],
                      "rchi2": [2],
                      "AIC": [3],
                      "BIC": [4]}
        for i, parameter_name in enumerate(model.parameters):
            self.index[parameter_name] = [1, 'x', i]

        # Spatial size of the data cube in pixels
        self.ny = data.shape[0]
        self.nx = data.shape[1]

        # Get the fit results
        self.result = [[None]*self.nx for i in range(self.ny)]
        for i in range(0, self.nx):
            for j in range(0, self.ny):
                observed_power = data[j, i, :]
                guess = self.model.guess(self.fn, observed_power, **kwargs)
                result = lnlike_model_fit.go(self.fn,
                                             observed_power,
                                             self.model.power,
                                             guess,
                                             self.fit_method)

                # Estimates of the quality of the fit to the data
                parameter_estimate = result['x']
                bestfit = self.model.power(parameter_estimate, self.fn)
                rhoj = lnlike_model_fit.rhoj(observed_power, bestfit)
                rchi2 = lnlike_model_fit.rchi2(1.0, self.dof, rhoj)
                aic = lnlike_model_fit.AIC(self.k, parameter_estimate, self.fn,
                                           observed_power, self.model.power)
                bic = lnlike_model_fit.BIC(self.k, parameter_estimate, self.fn,
                                           observed_power, self.model.power,
                                           self.n)

                # Store the final results
                self.result[j][i] = (guess, result, rchi2, aic, bic)


    def as_array(self, quantity):
        """
        Convert parts of the results nested list into a numpy array.

        :param quantity: a string that indicates which quantity you want
        :return: numpy array of size (ny, nx)
        """
        as_array = self._as_array(self.index[quantity])

        # Convert to an easier to use value if returning a model parameter.
        if quantity in self.model.parameters:
            return as_array * self.model.conversion[self.model.parameters.index(quantity)]
        else:
            return as_array

    def _as_array(self, indices):
        as_array = np.zeros((self.ny, self.nx))
        for i in range(0, self.nx):
            for j in range(0, self.ny):
                x = self.result[j][i]
                for index in indices:
                    x = x[index]
                as_array[j, i] = x
        return as_array

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
        p values, and a successful fit as defined by the fitting algorithm/

        :param p_value: Two element object that has the lower and upper p values
        which are used to calculate corresponding reduced chi-squared values.
        :return: A logical mask where mask = False indicates a GOOD fit,
        mask = True indicates a BAD fit.  This can be passed directly into
        numpy's masked array.
        """
        return self.good_rchi2_mask(p_value) * self.as_array("success")

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
