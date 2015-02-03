#
# Fit an arbitrary function with a model using maximum likelihood,
# assuming exponential distributions.
#
import numpy as np
import scipy.optimize as op

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
