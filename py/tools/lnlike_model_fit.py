#
# Fit an arbitrary function with a model using maximum likelihood,
# assuming exponential distributions.
#
import numpy as np
import scipy.optimize as op
from scipy.stats import gengamma

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
# Sample to Model Ratio (SMR) estimator
#
def rhoj(Sj, shatj):
    """
    Sample to Model Ratio (SMR) estimator
    Nita et al (2014), ApJ, 789, 152, eq. 5

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
    Goodness-of-fit estimator
    Nita et al (2014), ApJ, 789, 152, eq. 16

    :param m: number of spectra considered
    :param nu: degrees of freedom
    :param rhoj: sample to model ratio estimator
    :return: SMR estimator
    """
    return (m / (1.0 * nu)) * np.sum((1.0 - rhoj) ** 2)

#
# PDF of the goodness-of-fit estimator
#
#
def rchi2distrib(m, nu):
    """
    The PDF of rchi2 may be approximated by the analytical expression
    below.

    :param m: number of spectra considered
    :param nu: degrees of freedom
    :return: a frozen scipy stats function that represents the distribution
    of the data
    """
    a = 1.0 + 3.0 / (1.0 * m)
    b = nu / 2.0
    return gengamma(b / a, a / b)
