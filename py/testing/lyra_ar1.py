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
#from rnsimulation import SimplePowerLawSpectrumWithConstantBackground, TimeSeriesFromPowerSpectrum
import pickle
import sunpy
import datetime
from statsmodels.tsa.api import *

plt.ion() 

# Filename
filename = os.path.expanduser('~/Data/LYRA/example/5_Sep_2011_lyra_al_series.pickle')

# Load in the pickle file
lyra = pickle.load(open(filename, 'rb'))

# Get each time series
results = []
for ts in lyra:
    basetime = sunpy.time.parse_time(ts.SampleTimes.basetime)
    index = [basetime + datetime.timedelta(0, n) for n in ts.SampleTimes.time]
    model = AR(ts.data, dates=index)
    results.append(model.fit(1))
