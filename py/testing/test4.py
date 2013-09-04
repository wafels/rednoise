from __future__ import absolute_import
"""
Co-align a set of maps and make sure the
"""

# Test 4: Randomly subsample the spatial positions
import scipy 
import numpy as np
from matplotlib import pyplot as plt
import pymc
from rnfit2 import Do_MCMC
import sunpy
import pickle
import os
from pymcmodels import single_power_law_with_constant
from cubetools import get_datacube
from rnsimulation import TimeSeries
import numpy as np
import os
from rnfit2 import Do_MCMC
from rnsimulation import TimeSeries, SimplePowerLawSpectrumWithConstantBackground, TimeSeriesFromPowerSpectrum
from matplotlib import pyplot as plt
from pymcmodels import single_power_law_with_constant
plt.ion()


# Directory where the data is
if True:
    wave = '171'
    dir = os.path.expanduser('~/Data/AIA_Data/')
    sol = 'SOL2011-04-30T21-45-49L061C108/'
    #sol = 'SOL2011-05-09T22-28-13L001C055/'
    directory = dir + sol + wave + '/'# Save location for pickle files
    rootdir = os.path.expanduser('~/ts/pickle/jul30/') + sol + wave + '/'
else:
    directory = os.path.expanduser('~/Data/oscillations/mcateer/outgoing3/QS_E.sav')
    rootdir = os.path.expanduser('~/Desktop/')

directory = os.path.expanduser('~/Data/AIA_Data/rn1/1/')

print('Loading ' + directory)


# make the directory
if not os.path.exists(rootdir):
    os.makedirs(rootdir)

# Load in the data
dc = get_datacube(directory)
ny = dc.shape[0]
nx = dc.shape[1]
nt = dc.shape[2]


dt = 12
# Random subsamples
# Number of samples
nsample = 1# np.min([nx * ny / 100, 250])
# Seed to ensure repeatability
np.random.seed(seed=2)
# Unique random locations
isunique = False
while isunique is False:
    rand_x = np.random.randint(0, high=nx, size=nsample)
    rand_y = np.random.randint(0, high=ny, size=nsample)
    locations = zip(rand_y, rand_x)
    if len(locations) == len(list(set(locations))):
        isunique = True

# Result # 1 - add up all the emission and do the analysis on the full FOV
full_ts = np.sum(dc, axis=(0, 1))


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

full_ts = fix_nonfinite(full_ts)
t = dt * np.arange(0, len(full_ts))
ts = TimeSeries(t, full_ts)



filename = rootdir + 'test4.full_ts.pickle'
analysis = Do_MCMC([ts]).okgo(single_power_law_with_constant, iter=50000, burn=10000, thin=5, progress_bar=False)


# Do the posterior predictive statistics to measure GOF
def vaughan_2010_T_R(iobs, S):
    """Vaughan, 2010, MNRAS, 402, 307. Eq. 15.
    Returns a test statistic measuring the difference between
    the observed power spectrum iobs and another power spectrum S."""
    return np.max(2 * iobs / S)


def vaughan_2010_T_SSE(iobs, S):
    """Vaughan, 2010, MNRAS, 402, 307. Eq. 21.
    Returns a test statistic measuring the difference between
    the observed power spectrum iobs and another power spectrum S."""
    return np.sum(((iobs - S) / S) ** 2)


def vaughan_2010_T_LRT(logp_model1, logp_model2):
    """Vaughan, 2010, MNRAS, 402, 307. Eq. 22.
    Returns a test statistic used to compare nested models."""
    return -2 * (logp_model1 - logp_model2)



def posterior_predictive_distribution(iobs, fit_results,
                                      nsample=1000,
                                      statistic='vaughan_2010_T_R',
                                      ):
    # Storage for the distribution results
    distribution = []

    # Sample from the Bayes posterior for the fit, generate a spectrum,
    # and calculate the test statistic
    for i in range(0, nsample):
        # get a random sample from the posterior
        r = np.random.randint(0, fit_results["power_law_index"].shape[0])
        norm = fit_results["power_law_norm"][r]
        index = fit_results["power_law_index"][r]
        background = fit_results["background"][r]

        # Define some simulated time series data with the required spectral
        # properties
        pls = SimplePowerLawSpectrumWithConstantBackground([norm, index, background], nt=nt, dt=dt)
        data = TimeSeriesFromPowerSpectrum(pls).sample

        # Create a TimeSeries object from the simulated data
        ts = TimeSeries(dt * np.arange(0, nt), data)

        # Get the simulated data's power spectrum
        S = ts.PowerSpectrum.Npower

        # Calculate the required discrepancy statistic
        if statistic == 'vaughan_2010_T_R':
            value = vaughan_2010_T_R(iobs, S)
        if statistic == 'vaughan_2010_T_SSE':
            value = vaughan_2010_T_R(iobs, S)
        distribution.append(value)

    return np.array(distribution)


# Get the MCMC results from the analysis
fit_results = analysis.results[0]["samples"]

# Get the MAP values
mp = analysis.results[0]["mp"]

# Get the list of variable names
l = str(list(mp.variables)[0].__name__)

# Best fit spectrum
mean = analysis.results[0]["mean"]
std = analysis.results[0]["std"]
best_fit_power_spectrum = SimplePowerLawSpectrumWithConstantBackground([mp.power_law_norm.value, mp.power_law_index.value, mp.background.value], nt=nt, dt=dt).power()

bfps1 = best_fit_power_spectrum / np.mean(best_fit_power_spectrum)

bfps2 = bfps1 / np.std(bfps1)

# Normalized observed power spectrum
iobs = ts.PowerSpectrum.Npower

plt.figure(1)
plt.loglog(ts.PowerSpectrum.frequencies.positive, iobs)
plt.loglog(ts.PowerSpectrum.frequencies.positive, bfps2)
plt.show()

# Calculate the posterior predictive distribution
x = posterior_predictive_distribution(iobs, fit_results, nsample=5000)

# Calculate the discrepancy statistic
value = vaughan_2010_T_R(iobs, bfps2)

plt.figure(2)
plt.hist(x, bins=100, range=[x.min(), x.min() + 100 * value])
plt.axvline(value)
plt.show()




"""
# Result 2 - add up the time series from all the randomly selected locations
# and do a fit
filename = rootdir + 'test4.rand_ts.pickle'
rand_ts = np.zeros(shape=nt)
for loc in locations:
    rand_ts = rand_ts + dc[loc[0], loc[1], :].flatten()
ts = TimeSeries(t, rand_ts)
rand_res = Do_MCMC([ts]).okgo(single_power_law_with_constant, iter=50000, burn=10000, thin=5, progress_bar=False)

# Result 3 - do all the randomly selected pixels one by one
filename = rootdir + 'test4.rand_locations.pickle'
tss = []
for loc in locations:
    tss.append(TimeSeries(t, dc[loc[0], loc[1], :].flatten()))
zall = Do_MCMC(tss).okgo(single_power_law_with_constant, iter=50000, burn=10000, thin=5, progress_bar=False)
"""