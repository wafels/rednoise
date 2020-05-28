#
# Power spectrum tools
#
import numpy as np
from scipy.stats import norm
from astropy.modeling.fitting import _fitter_to_model_params


#
# Assume a power law in power spectrum - Use a Bayesian marginal distribution
# to calculate the probability that the power spectrum has a power law index
# 'n'
#


def bayeslogprob(f, I, n, m):
    """
    Return the log of the marginalized Bayesian posterior of an observed
    Fourier power spectra fit with a model spectrum Af^{-n}, where A is a
    normalization constant, f is a normalized frequency, and n is the power law
    index at which the probability is calculated. The marginal probability is
    calculated using a prior p(A) ~ A^{m}.

    The function returns log[p(n)] give.  The most likely value of 'n' is the
    maximum value of p(n).

    f : normalized frequencies
    I : Fourier power spectrum
    n : power law index of the power spectrum
    m : power law index of the prior p(A) ~ A^{m}
    """
    N = len(f)
    term1 = n * np.sum(np.log(f))
    term2 = (N - m - 1) * np.log(np.sum(I * f ** n))
    return term1 - term2


#
# Find the most likely power law index given a prior on the amplitude of the
# power spectrum
#
def most_probable_power_law_index(f, I, m, n):
    blp = np.zeros_like(n)
    for inn, nn in enumerate(n):
        blp[inn] = bayeslogprob(f, I, nn, m)
    return n[np.argmax(blp)]


#
# Some integrals
#
def gaussian_component(b, sigma, c, mu0):
    """
    The integral from mu0 to infinity of the gaussian component used
    in the power spectral analysis.

    :param b: amplitude of the gaussian component used in the power spectral
    analysis
    :param sigma: width of the gaussian component used in the power spectral
    analysis
    :param c: center of the gaussian component used in the power spectral
    analysis
    :param mu0: lower end of normalized frequency range over which the
    integral is calculated.  The upper limit is set to infinity.
    :return: the integral from mu0 to infinity of the gaussian component used
    in the power spectral analysis
    """
    x = (np.log(mu0) - c) / sigma
    cdf = norm.cdf(x)
    return b * np.sqrt(2*np.pi) * sigma * (1.0 - cdf)


def power_law_component(a, n, mu0):
    """
    The integral from mu0 to infinity of the power law component used
    in the power spectral analysis.

    :param a: amplitude of the power law component
    :param n: power law index
    :param mu0: lower end of normalized frequency range over which the
    integral is calculated.  The upper limit is set to infinity.
    :return: the integral from mu0 to infinity of the power law component used
    in the power spectral analysis.
    """
    return (a / (n + 1.0)) * mu0 ** (1.0-n)


# Create an array of simulated Fourier powers
def create_simulated_power_spectra(nx, ny, observation_model, model_parameters, frequencies):
    d = np.zeros([nx, ny, len(frequencies)])
    for i in range(0, nx):
        for j in range(0, ny):
            # This section will be replaced with a section that reads observed power spectra
            # Set the model parameters
            _fitter_to_model_params(observation_model,
                                    [model_parameters[0], model_parameters[1], model_parameters[2]])
            # Create the true data
            psd_shape = observation_model(frequencies)

            # Now randomize the true data and store it in an iterable
            d[i, j, :] = psd_shape * np.random.chisquare(2, size=psd_shape.shape[0]) / 2.0
    return d

