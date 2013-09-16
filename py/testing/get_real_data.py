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
# plt.ion()

#  _____________________________________________________________________________
# Main directory where the data is
maindir = os.path.expanduser('~/Data/AIA_Data/rn4/')

# Which wavelength to look at
wave = '304'

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

# Result # 1 - add up all the emission and do the analysis on the full FOV
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

iobs = iobs / (1.0 * nx * ny)
ts.label = 'emission (AIA ' + wave + ')'
ts.units = 'counts'

# Get the normalized power and the positive frequencies
iobs = ts.PowerSpectrum.Npower
this = ([ts.PowerSpectrum.frequencies.positive, iobs],)
