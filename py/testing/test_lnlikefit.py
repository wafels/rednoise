import numpy as np
from scipy.stats import expon
import rnspectralmodels
import lnlike_model_fit

#
# set up some test data
#
nt = 1800

true_values = [np.log(1.0), 2.2]

this_model = rnspectralmodels.power_law

# Now set up some simulated data and do the fit
freqs = 1.0 * np.arange(1, 899)
data = np.zeros_like(freqs)
S = this_model(true_values, freqs)
for i, s in enumerate(S):
    data[i] = expon.rvs(scale=s)

for z in range(0, 2):

    # Initial guess
    initial_guess = [np.log(data[0]), 1.5 - 0.5*z, np.log(data[-1])]

    #
    k = len(initial_guess)
    n = len(freqs)

    # Do the fit
    result = lnlike_model_fit.go(freqs, data, this_model, initial_guess, "Nelder-Mead")

    # Calculate Nita et al reduced chi-squared value
    bestfit = this_model(result['x'], freqs)

    # Sample to model ratio
    rhoj = lnlike_model_fit.rhoj(data, bestfit)

    # Nita et al (2014) value of the reduced chi-squared
    rchi2 = lnlike_model_fit.rchi2(1, n - k - 1, rhoj)

    # AIC
    aic = lnlike_model_fit.AIC(k, result['x'], freqs, data, this_model)

    # BIC
    bic = lnlike_model_fit.BIC(k, result['x'], freqs, data, this_model, n)

    print aic, bic, result['x']


