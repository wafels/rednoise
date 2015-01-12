#
# Simple test of the log likelihood fitting
#
import numpy as np
import scipy.optimize as op
from scipy.stats import expon
import matplotlib.pyplot as plt


def lnlike(variables, x, y, model_function):
    """
    Log likelihood of the data given a model.  Assumes that the data is
    exponentially distributed.  Can be used to fit Fourier power spectra.
    :param variables:
    :param data:
    :param frequencies:
    :param model_function:
    :return:
    """
    model = model_function(variables, x)
    return -np.sum(np.log(model)) - np.sum(y / model)


def power_law(v, f):
    return v[0] * f ** -v[1]

initial_guess = [10.0, 1.7]
f = 1.0 * np.arange(1, 899)
I = np.zeros_like(f)
S = power_law(initial_guess, f)
for i, s in enumerate(S):
    I[i] = expon.rvs(scale=s)

nll = lambda *args: -lnlike(*args)
args = (f, I, power_law)
result = op.minimize(nll, initial_guess, args=args, method='Nelder-Mead')
a, n = result["x"]


plt.figure(1)
plt.loglog(f, S, label='true : n = %f' % (initial_guess[1]))
plt.loglog(f, I, label='noisy')
plt.loglog(f, args[2](result["x"], f), label='fit: n = %f' % (n))
plt.xlabel('normalized frequency')
plt.ylabel('power')
plt.legend(framealpha=0.5)
plt.show()