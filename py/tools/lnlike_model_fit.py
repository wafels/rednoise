#
# Fit an arbitrary function with a model using maximum likelihood,
# assuming exponential distributions.
#
import numpy as np
import scipy.optimize as op
from scipy.stats import gamma

#
# Log likelihood function.  In this case we want the product of exponential
# distributions.
#
def lnlike(variables, x, y, model_function):
    """
    Log likelihood of the data given a model.  Assumes that the data is
    exponentially distributed.  Can be used to fit Fourier power spectra.
    :param variables: array like, variables used by model_function
    :param x: the independent variable (most often normalized frequency)
    :param y: the dependent variable (observed power spectrum)
    :param model_function: the model that we are using to fit the power
    spectrum
    :return: the log likelihood of the data given the model.
    """
    model = model_function(variables, x)
    return -np.sum(np.log(model)) - np.sum(y / model)


#
# Fit the input model to the data.
#
def go(freqs, data, model_function, initial_guess, method):
    nll = lambda *args: -lnlike(*args)
    args = (freqs, data, model_function)
    return op.minimize(nll, initial_guess, args=args, method=method)


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
# The code below refers to equations in Nita et al (2014), ApJ, 789, 152
#
#
# Sample to Model Ratio (SMR) estimator
#
def rhoj(Sj, shatj):
    """
    Sample to Model Ratio (SMR) estimator (Eq. 5)

    :param Sj: random variables (data)
    :param shatj: best estimate of the model
    :return: SMR estimator
    """
    return Sj / shatj


#
# Goodness-of-fit estimator
#
def rchi2(m, nu, rhoj):
    """
    Goodness-of-fit estimator (Eq. 16)

    :param m: number of spectra considered
    :param nu: degrees of freedom
    :param rhoj: sample to model ratio estimator
    :return: SMR estimator
    """
    return (m / (1.0 * nu)) * np.sum((1.0 - rhoj) ** 2)

#
# PDF of the goodness-of-fit estimator (Eq. 17)
#
def rchi2distrib(m, nu):
    """
    The distribution of rchi2 may be approximated by the analytical expression
    below.  Comparing Eq. (2) with the implementation in scipy stats we find
    the following equivalencies:
    k (Nita parameter) = a (scipy stats parameter)
    theta (Nita parameter) = 1 / lambda (scipy stats value)


    :param m: number of spectra considered
    :param nu: degrees of freedom
    :return: a frozen scipy stats function that represents the distribution
    of the data
    """
    # Calculate the gamma function parameter values as expressed in Eq. 17
    k = 1.0 + 3.0 / (1.0 * m)
    theta = nu / 2.0
    #
    return gamma(k, scale=theta, loc=0.0)
