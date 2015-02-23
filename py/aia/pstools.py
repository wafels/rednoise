#
# Power spectrum tools
#
import numpy as np


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
    return n[np.argmin(blp)]

#
# Generate an initial guess to the log likelihood fit
#
def generate_initial_guess(model_name, f, p):
    if model_name == 'power law':
        index_estimate = most_probable_power_law_index(f, p, 0.0, np.arange(0.0, 4.0, 0.01))
        initial_guess = [np.log(p[0]), index_estimate]

    if model_name == 'power law with constant':
        index_estimate = most_probable_power_law_index(f, p, 0.0, np.arange(0.0, 4.0, 0.01))
        initial_guess = [np.log(p[0]), index_estimate, np.log(p[-1])]


    return initial_guess