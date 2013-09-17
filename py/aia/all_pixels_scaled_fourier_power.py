"""
Specifies a directory of AIA files and calculates the scaled power law for
all the pixels in the derotated datacube.
"""
# Test 6: Posterior predictive checking
import numpy as np
import os
from matplotlib import pyplot as plt
#import sunpy
#import pymc
import tsutils
from rnfit2 import Do_MCMC, rnsave
import ppcheck2
from pymcmodels import single_power_law_with_constant
from cubetools import get_datacube
from timeseries import TimeSeries
from rnsimulation import SimplePowerLawSpectrumWithConstantBackground, TimeSeriesFromPowerSpectrum


# Main directory where the data is
maindir = os.path.expanduser('~/Data/AIA_Data/SOL2011-04-30T21-45-49L061C108')

# Which wavelength to look at
wave = '171'

# Construct the directory
directory = os.path.join(maindir, wave)

# Load in the data
print('Loading data from ' + directory)
dc = get_datacube(directory)
ny = dc.shape[0]
nx = dc.shape[1]
nt = dc.shape[2]
# Create a time series object
dt = 12.0
t = dt * np.arange(0, nt)
tsdummy = TimeSeries(t, t)
iobs = np.zeros(tsdummy.PowerSpectrum.Npower.shape)
nposfreq = len(iobs)

# Result # 1 - add up all the emission and do the analysis on the full FOV
# Also, make a histogram of all the power spectra to get an idea of the
# varition present

# number of histogram bins
bins = 50

# storage
pwr = np.zeros((ny, nx, nposfreq))

full_ts = np.zeros((nt))
for i in range(0, nx):
    for j in range(0, ny):
        d = dc[j, i, :].flatten()
        # Fix the data for any non-finite entries
        d = tsutils.fix_nonfinite(d)
        d = d - np.mean(d)
        d = d / np.std(d)
        ts = TimeSeries(t, d)
        iobs = iobs + ts.PowerSpectrum.Npower
        pwr[j, i, :] = np.log(ts.PowerSpectrum.Npower)

# Average power in units of estimated standard deviation
iobs = iobs / (1.0 * nx * ny)

# Create the histogram of all the powers
hpwr = np.zeros((nposfreq, bins))
for f in range(0, nposfreq):
    h = np.histogram(pwr[:, :, f], bins=bins, range=[pwr.min(), pwr.max()])
    hpwr[f, :] = h[0] / (1.0 * h[0].max())

