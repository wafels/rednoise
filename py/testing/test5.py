from __future__ import absolute_import
"""
Co-align a set of maps and make sure the
"""

# Test 4: Randomly subsample the spatial positions
import numpy as np 
from matplotlib import pyplot as plt
import pymc
from rnfit2 import Do_MCMC
import sunpy
import pickle
import os
import ppcheck
from pymcmodels import single_power_law_with_constant
from cubetools import get_datacube
from rnsimulation import TimeSeries
import numpy as np
import os
from rnfit2 import Do_MCMC
from rnsimulation import TimeSeries, SimplePowerLawSpectrumWithConstantBackground, TimeSeriesFromPowerSpectrum
from matplotlib import pyplot as plt
from pymcmodels import single_power_law_with_constant
import ppcheck2
plt.ion()

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def fix_nonfinite(data):
    """
    Finds all the nonfinite regions in the data and replaces them with a simple
    linear interpolation.
    """
    good_indexes = np.isfinite(data)
    bad_indexes = np.logical_not(good_indexes)
    good_data = data[good_indexes]
    interpolated = np.interp(bad_indexes.nonzero()[0], good_indexes.nonzero()[0], good_data)
    data[bad_indexes] = interpolated
    return data

# -----------------------------------------------------------------------------
# Load the data
# -----------------------------------------------------------------------------
directory = os.path.expanduser('~/Data/AIA_Data/rn1/1/')
rootdir = os.path.expanduser('~/Desktop/')
print('Loading ' + directory)

# make the results directory
if not os.path.exists(rootdir):
    os.makedirs(rootdir)

# Load in the data
dc = get_datacube(directory)
ny = dc.shape[0]
nx = dc.shape[1]
nt = dc.shape[2]

# Force the cadence to be the AIA cadence
dt = 12.0

# Get the location of some random located samples
nrandom_location = 1

# Generate unique random locations
isunique = False
while isunique is False:
    rand_x = np.random.randint(0, high=nx, size=nrandom_location)
    rand_y = np.random.randint(0, high=ny, size=nrandom_location)
    locations = zip(rand_y, rand_x)
    if len(locations) == len(list(set(locations))):
        isunique = True

# Time Series 1 - add up all the emission and do the analysis on the full FOV
full_dc = fix_nonfinite(np.sum(dc, axis=(0, 1)))
t = dt * np.arange(0, len(full_dc))
ts = TimeSeries(t, full_dc)

# Get the normalized observed power spectrum and the positive frequencies
iobs = ts.PowerSpectrum.Npower
this = ([ts.PowerSpectrum.frequencies.positive, iobs],)

# -----------------------------------------------------------------------------
# Analyze using MCMC
# -----------------------------------------------------------------------------
analysis = Do_MCMC(this).okgo(single_power_law_with_constant, iter=50000, burn=10000, thin=5, progress_bar=False)

# Get the MAP values
mp = analysis.results[0]["mp"]

# Get the full MCMC object
M = analysis.results[0]["M"]

# Get the list of variable names
l = str(list(mp.variables)[0].__name__)

# Best fit spectrum
best_fit_power_spectrum = SimplePowerLawSpectrumWithConstantBackground([mp.power_law_norm.value, mp.power_law_index.value, mp.background.value], nt=nt, dt=dt).power()

plt.figure(1)
plt.loglog(ts.PowerSpectrum.frequencies.positive, iobs, label="normalized observed power spectrum")
plt.loglog(ts.PowerSpectrum.frequencies.positive, best_fit_power_spectrum, label="best fit")
plt.axvline(1.0 / 300.0, color='k', linestyle='--', label='5 mins')
plt.axvline(1.0 / 180.0, color='k', linestyle=':', label='3 mins')
plt.legend(fontsize=10, loc=3)
plt.show()


# -----------------------------------------------------------------------------
# Now do the posterior predictive check
# -----------------------------------------------------------------------------
statistic = ('vaughan_2010_T_R', 'vaughan_2010_T_SSE')
value = {}
for k in statistic:
    value[k] = ppcheck2.calculate_statistic(k, iobs, best_fit_power_spectrum)

distribution = ppcheck2.posterior_predictive_distribution(ts,
                                                          M,
                                                          nsample=1000,
                                                          statistic=statistic,
                                                          verbose=True)
