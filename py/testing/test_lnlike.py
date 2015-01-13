#
# Simple test of the log likelihood fitting
#
import numpy as np
import scipy.optimize as op
from scipy.stats import expon
import matplotlib.pyplot as plt
plt.ion()

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

# Define some models
def power_law(v, f):
    return v[0] * f ** -v[1]

def power_law_constant(v, f):
    return np.exp(v[0]) * f ** -v[1] + np.exp(v[2])

# pick a model and some parameter values
this_model = power_law_constant
true_values = [np.log(1.0), 2.2, np.log(0.00001)]

# Now set up some simulated data and do the fit
f = 1.0 * np.arange(1, 899)
I = np.zeros_like(f)
S = this_model(true_values, f)
for i, s in enumerate(S):
    I[i] = expon.rvs(scale=s)

# Initial guess
initial_guess = [np.log(I[0]), 1.5, np.log(I[-1])]

nll = lambda *args: -lnlike(*args)
args = (f, I, this_model)

plt.figure(1)
plt.loglog(f, S, label='true : n = %f' % (true_values[1]))
plt.loglog(f, I, label='noisy')

methods = ['Nelder-Mead', 'Powell', 'CG', 'BFGS']
for method in methods:
    result = op.minimize(nll, initial_guess, args=args, method=method)
    fitvalues = result['x']
    plt.loglog(f, args[2](result["x"], f), label='fit (%s): n = %f' % (method, fitvalues[1]))

plt.xlabel('normalized frequency')
plt.ylabel('power')
plt.legend(framealpha=0.5, loc=3)
plt.show()