"""
Specifies a directory of AIA files and calculates the scaled power law for
all the pixels in the derotated datacube.
"""
# Test 6: Posterior predictive checking
import numpy as np
from matplotlib import pyplot as plt
#import sunpy
#import pymc
import tsutils
from rnfit2 import Do_MCMC, rnsave
#import ppcheck2
from pymcmodels import single_power_law_with_constant_not_normalized
import os
from timeseries import TimeSeries
from scipy.optimize import curve_fit
from scipy.io import readsav
from cubetools import get_datacube
from rnsimulation import SimplePowerLawSpectrumWithConstantBackground, TimeSeriesFromPowerSpectrum
import aia_specific
import scipy.stats as stats
plt.ion()

wave = '171'
dc, location = aia_specific.rn4(wave, location='~/Data/AIA/shutdownfun', derotate=False)


#wave = '171'
#dc, location = aia_specific.rn4(wave, Xrange=[-201, -1])

# Get some properties of the datacube
ny = dc.shape[0]
nx = dc.shape[1]
nt = dc.shape[2]

# Create a time series object
dt = 12.0
t = dt * np.arange(0, nt)
tsdummy = TimeSeries(t, t)
iobs = np.zeros(tsdummy.PowerSpectrum.Npower.shape)
logiobs = np.zeros(tsdummy.PowerSpectrum.Npower.shape)
nposfreq = len(iobs)

# Result # 1 - add up all the emission and do the analysis on the full FOV
# Also, make a histogram of all the power spectra to get an idea of the
# varition present

# storage
full_data = np.zeros((nt))
for i in range(0, nx):
    for j in range(0, ny):
        d = dc[j, i, :].flatten()
        # Fix the data for any non-finite entries
        d = tsutils.fix_nonfinite(d)

        # Sum up all the data
        full_data = full_data + d

full_data = tsutils.fix_nonfinite(dc[30, 110, :])

# Average emission over all the data
full_data = full_data #/ (1.0 * nx * ny)

# Create a time series object
full_ts = TimeSeries(t, full_data)

# Save the time-series
# Set up where to save the data, and the file type/
save = rnsave(root='~/ts/pickle',
              description='all_aia',
              filetype='pickle')
save.ts(full_ts)


freqs = full_ts.PowerSpectrum.frequencies.positive / full_ts.PowerSpectrum.frequencies.positive[0]

# Form the input for the MCMC algorithm.
iobs = full_ts.ppower
this = ([full_ts.pfreq, iobs],)


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

# -----------------------------------------------------------------------------
# Analyze using MCMC
# -----------------------------------------------------------------------------
analysis = Do_MCMC(this).okgo(single_power_law_with_constant_not_normalized,
                              estimate=estimate,
                              iter=50000,
                              burn=10000,
                              thin=5,
                              progress_bar=False)
# Get the MAP values
mp = analysis.mp
best_fit_power_spectrum = SimplePowerLawSpectrumWithConstantBackground([mp.power_law_norm.value, mp.power_law_index.value, mp.background.value], nt=nt, dt=dt).power()
r = iobs / best_fit_power_spectrum

h, xh = np.histogram(r, bins=20)
h = h / (1.0 * np.sum(h))
x = 0.5*(xh[0:-1] + xh[1:])
plt.plot(x, h)
xx = 0.01 * np.arange(0, 300)
plt.plot(xx, stats.chi2.pdf(xx, 2))
