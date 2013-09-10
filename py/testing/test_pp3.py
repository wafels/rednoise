"""
The purpose of this program is to analyze test data of a given power law index
and duration and find the probability distribution of measured power law
indices.
"""

import numpy as np
from rnfit2 import Do_MCMC
from rnsimulation import TimeSeries, SimplePowerLawSpectrumWithConstantBackground, TimeSeriesFromPowerSpectrum
from matplotlib import pyplot as plt
from pymcmodels import single_power_law_with_constant
import ppcheck2

# interactive mode
plt.ion()

# _____________________________________________________________________________
# Data goes here
# -----------------------------------------------------------------------------
# Create some fake data
# -----------------------------------------------------------------------------
dt = 12.0
nt = 300
np.random.seed(seed=1)
pls1 = SimplePowerLawSpectrumWithConstantBackground([10.0, 2.0, -5.0],
                                                    nt=nt,
                                                    dt=dt)
data = TimeSeriesFromPowerSpectrum(pls1).sample
t = dt * np.arange(0, nt)
amplitude = 0.0
data = data + amplitude * (data.max() - data.min()) * np.sin(2 * np.pi * t / 300.0)

# Create a time series object
ts = TimeSeries(t, data)

# Get the normalized power and the positive frequencies
iobs = ts.PowerSpectrum.Npower
this = ([ts.PowerSpectrum.frequencies.positive, iobs],)
# _____________________________________________________________________________
# -----------------------------------------------------------------------------
# Analyze using MCMC
# -----------------------------------------------------------------------------
analysis = Do_MCMC(this).okgo(single_power_law_with_constant, iter=50000, burn=10000, thin=5, progress_bar=False)

# Get the MAP values
mp = analysis.results[0]["mp"]

# Get the full MCMC object
M = analysis.results[0]["M"]

# Get the list of variable names
#l = str(list(mp.variables)[0].__name__)

# Best fit spectrum
best_fit_power_spectrum = SimplePowerLawSpectrumWithConstantBackground([mp.power_law_norm.value, mp.power_law_index.value, mp.background.value], nt=nt, dt=dt).power()

# -----------------------------------------------------------------------------
# Now do the posterior predictive check
# -----------------------------------------------------------------------------
statistic = ('vaughan_2010_T_R', 'vaughan_2010_T_SSE')
nsample = 1000
value = {}
for k in statistic:
    value[k] = ppcheck2.calculate_statistic(k, iobs, best_fit_power_spectrum)

distribution = ppcheck2.posterior_predictive_distribution(ts,
                                                          M,
                                                          nsample=nsample,
                                                          statistic=statistic,
                                                          verbose=True)

# -----------------------------------------------------------------------------
# Summary plots
# -----------------------------------------------------------------------------
# Best fit
plt.figure()
plt.loglog(ts.PowerSpectrum.frequencies.positive, iobs, label="normalized observed power spectrum")
plt.loglog(ts.PowerSpectrum.frequencies.positive, best_fit_power_spectrum, label="best fit")
plt.axvline(1.0 / 300.0, color='k', linestyle='--', label='5 mins')
plt.axvline(1.0 / 180.0, color='k', linestyle=':', label='3 mins')
plt.legend(fontsize=10, loc=3)
plt.show()

# Discrepancy statistics
for k in statistic:
    v = value[k]
    d = distribution[k]
    pvalue = np.sum(distribution > v) / (1.0 * nsample)

    plt.figure()
    h = plt.hist(d, bins=20)
    plt.axvline(value, color='k')
    plt.xlabel("statistic value")
    plt.ylabel("Number found (%i samples)" % (nsample))
    plt.title('Statistic: ' + k)
    plt.text(v, np.max(h[0]), "p = %f" % (pvalue))
    plt.show()
