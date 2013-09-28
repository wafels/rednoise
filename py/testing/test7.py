from __future__ import absolute_import
"""
Co-align a set of maps and make sure the
"""

# Test 6: Posterior predictive checking
import numpy as np
#import os
from matplotlib import pyplot as plt
#import sunpy
#import pymc
#import tsutils
from rnfit2 import Do_MCMC, rnsave
import ppcheck2
from pymcmodels import single_power_law_with_constant_not_normalized
#from cubetools import get_datacube
from timeseries import TimeSeries
from rnsimulation import SimplePowerLawSpectrumWithConstantBackground, TimeSeriesFromPowerSpectrum
#import pickle

# matplotlib interactive mode
# plt.ion()

# _____________________________________________________________________________

if True:
    print('Simulated data')
    dt = 12.0
    nt = 300
    seed = 1
    np.random.seed(seed)
    pls1 = SimplePowerLawSpectrumWithConstantBackground([10.0, 2.0, -5.0],
                                                        nt=nt,
                                                        dt=dt)
    data = TimeSeriesFromPowerSpectrum(pls1).sample
    t = dt * np.arange(0, nt)
    amplitude = 0.0
    data = 1000*data + amplitude * (data.max() - data.min()) * np.sin(2 * np.pi * t / 300.0)
    ts = TimeSeries(t, data)

# Save the time-series
# Set up where to save the data, and the file type/
save = rnsave(root='~/ts/pickle',
              description='simulated.' + str(amplitude),
              filetype='pickle')
save.ts(ts)

# Get the positive frequencies and the power at those frequencies
iobs = ts.PowerSpectrum.ppower
"""
# Normalize to the number of elements in the original time series.  The
# motivation for this is that if the original time-series is pure Gaussian
# white noise with mean zero and standard devation of 1, then the average of
# the power spectrum is 1.
iobs = iobs / (1.0 * nt)
"""
# Normalize the frequency so that the first element is equal to 1 and
# all higher frequencies are multiples of the first non-zero frequency.  This
# makes calculation slightly easier.
freqs = ts.PowerSpectrum.frequencies.positive / ts.PowerSpectrum.frequencies.positive[0]

# Form the input for the MCMC algorithm.
this = ([ts.PowerSpectrum.frequencies.positive, iobs],)


norm_estimate = np.zeros((3,))
norm_estimate[0] = iobs[0]
norm_estimate[1] = norm_estimate[0] / 1000.0
norm_estimate[2] = norm_estimate[0] * 1000.0

background_estimate = np.zeros_like(norm_estimate)
background_estimate[0] = np.mean(iobs[-10:-1])
background_estimate[1] = background_estimate[0] / 1000.0
background_estimate[2] = background_estimate[0] * 1000.0

estimate = {"norm_estimate": norm_estimate,
            "background_estimate": background_estimate}

# _____________________________________________________________________________
# -----------------------------------------------------------------------------
# Analyze using MCMC
# -----------------------------------------------------------------------------
analysis = Do_MCMC(this).okgo(single_power_law_with_constant_not_normalized,
                              estimate=estimate,
                              seed=seed,
                              iter=50000,
                              burn=10000,
                              thin=5,
                              progress_bar=False,
                              db=save.filetype,
                              dbname=save.MCMC_filename)

# Get the MAP values
mp = analysis.mp

# Get the full MCMC object
M = analysis.M

# Save the MCMC output using the recommended save functionality
save.PyMC_MCMC(M)

# Save everything else
save.analysis_summary(analysis.results[0])


# Get the list of variable names
#l = str(list(mp.variables)[0].__name__)

# Best fit spectrum
best_fit_power_spectrum = SimplePowerLawSpectrumWithConstantBackground([mp.power_law_norm.value, mp.power_law_index.value, mp.background.value], nt=nt, dt=dt).power()
print mp.power_law_norm.value, mp.power_law_index.value, mp.background.value

# -----------------------------------------------------------------------------
# Now do the posterior predictive check - computationally time-consuming
# -----------------------------------------------------------------------------
statistic = ('vaughan_2010_T_R', 'vaughan_2010_T_SSE')
nsample = 10
value = {}
for k in statistic:
    value[k] = ppcheck2.calculate_statistic(k, iobs, best_fit_power_spectrum)

print value

distribution = ppcheck2.posterior_predictive_distribution(ts,
                                                          M, estimate,
                                                          nsample=nsample,
                                                          statistic=statistic,
                                                          verbose=True)
# Save the test statistic distribution information
save.posterior_predictive((value, distribution))

# -----------------------------------------------------------------------------
# Summary plots
# -----------------------------------------------------------------------------

# Plot the best fit
plt.figure()
plt.loglog(ts.PowerSpectrum.frequencies.positive, iobs, label="normalized observed power spectrum")
plt.loglog(ts.PowerSpectrum.frequencies.positive, best_fit_power_spectrum, label="best fit")
plt.axvline(1.0 / 300.0, color='k', linestyle='--', label='5 mins')
plt.axvline(1.0 / 180.0, color='k', linestyle=':', label='3 mins')
plt.legend(fontsize=10, loc=3)
plt.show()
"""
# Discrepancy statistics
for i, k in enumerate(statistic):
    v = value[k]
    d = distribution[k]
    pvalue = np.sum(d > v) / (1.0 * nsample)

    plt.figure(i)
    h = plt.hist(d, bins=20)
    plt.axvline(v, color='k')
    plt.xlabel("statistic value")
    plt.ylabel("Number found (%i samples)" % (nsample))
    plt.title('Statistic: ' + k)
    plt.text(v, np.max(h[0]), "p = %f" % (pvalue))
plt.show()
"""