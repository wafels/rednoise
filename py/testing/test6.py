from __future__ import absolute_import
"""
Co-align a set of maps and make sure the
"""

# Test 6: Posterior predictive checking
import numpy as np
import os
from matplotlib import pyplot as plt
import sunpy
import pymc
import tsutils
from rnfit2 import Do_MCMC
import ppcheck2
from pymcmodels import single_power_law_with_constant
from cubetools import get_datacube
from timeseries import TimeSeries
from rnsimulation import SimplePowerLawSpectrumWithConstantBackground

# matplotlib interactive mode
plt.ion()

# _____________________________________________________________________________
# Main directory where the data is
maindir = os.path.expanduser('~/Data/AIA_Data/SOL2011-04-30T21-45-49L061C108')

# Which wavelength to look at
wave = '193'

# Construct the directory
directory = os.path.join(maindir, wave)

# Load in the data
print('Loading data from ' + directory)
dc = get_datacube(directory)
#ny = dc.shape[0]
#nx = dc.shape[1]
nt = dc.shape[2]

# Result # 1 - add up all the emission and do the analysis on the full FOV
full_ts = np.sum(dc, axis=(0, 1))

# Fix the data for any non-finite entries
full_ts = tsutils.fix_nonfinite(full_ts)

# Create a time series object
dt = 12.0
t = dt * np.arange(0, len(full_ts))
ts = TimeSeries(t, full_ts)
ts.label = 'emission (AIA ' + wave + ')'
ts.units = 'counts'

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
# Now do the posterior predictive check - expensive
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

# Plot the best fit
plt.figure()
plt.loglog(ts.PowerSpectrum.frequencies.positive, iobs, label="normalized observed power spectrum")
plt.loglog(ts.PowerSpectrum.frequencies.positive, best_fit_power_spectrum, label="best fit")
plt.axvline(1.0 / 300.0, color='k', linestyle='--', label='5 mins')
plt.axvline(1.0 / 180.0, color='k', linestyle=':', label='3 mins')
plt.legend(fontsize=10, loc=3)
plt.show()

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
