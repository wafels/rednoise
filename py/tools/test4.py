from __future__ import absolute_import
"""
Co-align a set of maps and make sure the
"""

# Test 4: Randomly subsample the spatial positions
import scipy 
import numpy as np
from matplotlib import pyplot as plt
import pymc
from cubetools import derotated_datacube_from_mapcube
from rnfit import Do_MCMC
import sunpy
import pickle
import os
from pymcmodels import single_power_law_with_constant

# Directory where the data is
wave = '193'
dir = os.path.expanduser('~/Data/AIA_Data/')
sol = 'SOL2011-04-30T21-45-49L061C108/'
#sol = 'SOL2011-05-09T22-28-13L001C055/'
directory = dir + sol + wave + '/'

print('Loading ' + directory)
# Save location for pickle files
rootdir = os.path.expanduser('~/ts/pickle/jul30/') + sol + wave + '/'
# make the directory
if not os.path.exists(rootdir):
    os.makedirs(rootdir)

# Get a mapcube
maps = sunpy.Map(directory, cube=True)
# Get the datacube
maps[0].peek()
dc = derotated_datacube_from_mapcube(maps)
ny = dc.shape[0]
nx = dc.shape[1]
nt = dc.shape[2]
# Result # 1 - add up all the emission and do the analysis on the full FOV
full_ts = np.sum(dc, axis=(0, 1))
filename = rootdir + 'test4.full_ts.pickle'
full_ts = Do_MCMC(full_ts, 12.0).okgo(single_power_law_with_constant, iter=50000, burn=10000, thin=5, progress_bar=False)

# Random subsamples
# Number of samples
nsample = nx * ny / 100
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

# Result 2 - add up the time series from all the randomly selected locations
# and do a fit
filename = rootdir + 'test4.rand_ts.pickle'
rand_ts = np.zeros(shape=nt)
for loc in locations:
    rand_ts = rand_ts + dc[loc[0], loc[1], :].flatten()
rand_ts = Do_MCMC(rand_ts, 12.0).okgo(single_power_law_with_constant, iter=50000, burn=10000, thin=5, progress_bar=False)

# Result 3 - do all the randomly selected pixels one by one
#filename = rootdir + 'test4.rand_locations.pickle'
#zall = Do_MCMC(dc, 12.0).okgo(single_power_law_with_constant, iter=50000, burn=10000, thin=5, progress_bar=False, locations=locations).save(filename=filename)

