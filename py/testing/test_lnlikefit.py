import numpy as np
from scipy.stats import expon
from tools.rnspectralmodels import power_law
import tools.lnlike_model_fit as lnlike_model_fit

#
# set up some test data
#
nt = 1800

true_values = [np.log(1.0), 2.2]

this_model = power_law

# Now set up some simulated data and do the fit
freqs = 1.0 * np.arange(1, 899)
data = np.zeros_like(freqs)
S = this_model(true_values, freqs)
for i, s in enumerate(S):
    data[i] = expon.rvs(scale=s)

# Initial guess
initial_guess = [np.log(data[0]), 1.5, np.log(data[-1])]

# Do the fit
result = lnlike_model_fit.go(freqs, data, this_model, initial_guess, "Nelder-Mead")

# Calculate Nita et al reduced chi-squared value
bestfit = this_model(result['x'], freqs)

rhoj = lnlike_model_fit.rhoj(data, bestfit)

rchi2 = lnlike_model_fit.rchi2(1, len(freqs) - len(initial_guess) - 1, rhoj)

