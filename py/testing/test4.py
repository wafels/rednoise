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
t = 12.0 * np.arange(0, len(full_ts))
ts = TimeSeries(t, full_ts)



filename = rootdir + 'test4.full_ts.pickle'
full_res = Do_MCMC([ts]).okgo(single_power_law_with_constant, iter=50000, burn=10000, thin=5, progress_bar=False)
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