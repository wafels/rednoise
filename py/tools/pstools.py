#
# Power spectrum tools
#
import numpy as np
import rnspectralmodels
import details_study as ds
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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
# Implement what to do with the isolated structure limits
#
def structure_location(estimate):
    if estimate > ds.structure_location_limits['hi']:
        return ds.structure_location_limits['hi']
    if estimate < ds.structure_location_limits['lo']:
        return ds.structure_location_limits['lo']
    return estimate


def background_spectrum_estimate(finput, p,
                                 amp_range=[0, 5],
                                 index_range=[0, 50],
                                 background_range=[-50, -1]):

    # Normalize the input frequency.
    f = finput / finput[0]

    log_amplitude_estimate = np.log(np.mean(p[amp_range[0]:amp_range[1]]))
    index_estimate = most_probable_power_law_index(f[index_range[0]:index_range[1]],
                                                   p[index_range[0]:index_range[1]],
                                                   0.0, np.arange(0.0, 4.0, 0.01))
    log_background_estimate = np.log(np.mean(p[background_range[0]:background_range[1]]))
    return log_amplitude_estimate, index_estimate, log_background_estimate
#
# Generate an initial guess to the log likelihood fit
#
def generate_initial_guess(model_name, finput, p,
                                 amp_range=[0, 5],
                                 index_range=[0, 50],
                                 background_range=[-50, -1]):

    # No initial guess
    initial_guess = None

    # Normalize the input frequency.
    f = finput / finput[0]

    # Generate some initial simple estimates to the power law component
    log_amplitude, index_estimate, log_background = background_spectrum_estimate(finput, p,
                                 amp_range=amp_range,
                                 index_range=index_range,
                                 background_range=background_range)
    background_spectrum = rnspectralmodels.power_law_with_constant([log_amplitude, index_estimate, log_background], f)

    """
    plt.loglog(f, p, label='data')
    plt.loglog(f, background_spectrum, label='background spectrum estimate', linestyle='--')
    plt.axvline(f[amp_range[0]], linestyle=':')
    plt.axvline(f[amp_range[1]], linestyle=':', label='amplitude range')
    plt.axvline(f[index_range[0]], linestyle=':', color='k')
    plt.axvline(f[index_range[1]], linestyle=':', color='k', label='index range')
    plt.axvline(f[background_range[0]], linestyle=':', color='g')
    plt.axvline(f[background_range[1]], linestyle=':', color='g', label='background range')
    """

    if model_name == 'power law':
        initial_guess = [log_amplitude, index_estimate]

    if model_name == 'power law with constant':
        initial_guess = [log_amplitude, index_estimate, log_background]

    if model_name == 'power law with constant and delta function':

        # Location of the biggest difference between the
        delta_location_index = np.argmax(p - background_spectrum)

        # Make sure the location of the delta function is within required limits
        delta_location = structure_location(f[delta_location_index])

        # Find the nearest index at the position of the delta function
        delta_location_index = np.argmin(np.abs(delta_location - f))

        # Define the estimate of delta function
        log_delta_amplitude = np.log((p - background_spectrum)[delta_location_index])

        # The logarithm of the delta function's location
        log_delta_location = np.log(f[delta_location])

        # Finalize the guess
        initial_guess = [log_amplitude, index_estimate, log_background, log_delta_amplitude, log_delta_location]

    if model_name == 'power law with constant and lognormal':

        # Difference between the data and the model
        diff0 = p - background_spectrum

        # Keep the positive parts only
        positive_index = diff0 > 0.0

        # Limit the fit to a specific frequency range
        f_lower_limit = 50.0
        f_upper_limit = 200.0
        f_above = f > f_lower_limit
        f_below = f < f_upper_limit

        # Which data to fit
        fit_here = positive_index * f_above * f_below

        # If there is sufficient positive data
        if len(fit_here) > 10:
            diff1 = diff0[fit_here]
            f1 = f[fit_here]
            amp = np.log(np.max(diff1))
            pos = np.log(f1[np.argmax(diff1)])
            initial_guess = [log_amplitude, index_estimate, log_background, amp, pos, 0.1]
        else:
            initial_guess = [log_amplitude, index_estimate, log_background,
                             -100.0,
                             0.5 * (ds.structure_location_limits['lo'].value +
                                    ds.structure_location_limits['hi'].value),
                             0.1]
        #plt.loglog(f, rnspectralmodels.power_law_with_constant_with_lognormal(initial_guess, f), label='overall estimate')
        #plt.loglog(f1, diff1, label='used to fit lognormal,')
    #plt.legend(framealpha=0.5, fontsize=10)
    #plt.show()
    return initial_guess


def eqn4_8_23_summand(t, alpha_e, gamma, nu):
    exponent = -alpha_e*(1 + gamma) + gamma + 1.0 + gamma
    numerator = t ** exponent
    denominator = 1.0 + (2.0 * np.pi * nu * t) ** 2.0
    return numerator / denominator


def eqn4_8_23(timescales, alpha_e, gamma, nu):
    pwr = np.zeros_like(nu, dtype=float)
    for t in timescales:
        pwr += eqn4_8_23_summand(t, alpha_e, gamma, nu)
    return pwr
