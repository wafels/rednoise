from __future__ import absolute_import
"""
Co-align a set of maps and make sure the
"""

# Test 4: Randomly subsample the spatial positions
import numpy as np
import pymcmodels
import pymc
import tssimulation
from rnfit import Do_MCMC
import os
from pymcmodels import single_power_law_with_constant

# where the output data will be stored
rootdir = os.path.expanduser('~/ts/pickle/test4_traditional_model/')

# Number of samples to consider
nsample = 321

# Properties of the time_series
nt = 300
dt = 12.0
ampmax = 2.0

n_poly_high = 4

# Create the array of fake data
dc = np.zeros(shape=(nsample, nt))
for i in range(0, nsample):
    print('Creating fake data number %i' % (i))
    # maximum possible amplitude is ampmax
    A_osc = np.random.uniform(low=0.0, high=np.sqrt(ampmax))
    B_osc = np.random.uniform(low=0.0, high=np.sqrt(ampmax))
    # Random frequency in the range of observations
    frequency = 1.0 / np.random.uniform(low=140, high=360)
    # pick a random polynomial order
    n_polynomial = np.random.randint(low=0, high=n_poly_high)
    # pick some random polynomial coefficients.  Note that we have to be
    # careful to ensure that the highest order polynomials do not dominate
    # the emission
    # polynomial coefficients
    polynomial = np.random.uniform(low=-10 * ampmax, high=10 * ampmax, size=n_polynomial + 1)
    # the effect of the cofficients is about order 1
    if n_polynomial != 0:
        polynomial = polynomial / ((nt * dt) ** np.arange(n_polynomial, -1, -1))
    # create the time series
    dc[i, :] = tssimulation.time_series(nt, dt, A_osc, B_osc, frequency, polynomial)

# Result # 1 - do each fake time series individually
filename = rootdir + 'test4_traditional_model.all_samples.' + str(n_poly_high) + '.pickle'
print('Saving output to ' + filename)
zall = Do_MCMC(dc, dt).okgo(single_power_law_with_constant, iter=50000, burn=10000, thin=5, progress_bar=False, locations=np.arange(0,nsample)).save(filename=filename)

# Result # 2 - add up all the emission and do the analysis on the full FOV
full_ts = np.sum(dc, axis=0)
filename = rootdir + 'test4.full_ts.' + str(n_poly_high) + '.pickle'
z = Do_MCMC(full_ts, dt).okgo(single_power_law_with_constant, iter=50000, burn=10000, thin=5, progress_bar=False)
