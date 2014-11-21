#
#
#
import numpy as np

import aia_specific
import pymcmodels2
import rnspectralmodels
from paper1 import sunday_name, prettyprint, log_10_product, indexplot
from paper1 import csv_timeseries_write, pkl_write, fix_nonfinite, fit_details, get_kde_most_probable
from aia_pymc_pwrlaws_helpers import *
import matplotlib.pyplot as plt
from timeseries import TimeSeries

# Reproducible

filename = '/Users/ireland/Documents/Talks/2014/DirSem/sunspot171.npy'
data = np.load(filename)
meandata = np.mean(data)
data = (data - meandata)/meandata

nt = len(data)
t = 12.0 * np.arange(0, nt)

ts = TimeSeries(t, data * np.hanning(nt))

pwr = ts.PowerSpectrum.ppower
pfreqs = ts.PowerSpectrum.frequencies.positive
pymcmodel0 = pymcmodels2.splwc_exp(pfreqs, pwr)
M0 = pymc.MCMC(pymcmodel0)

# Run the sampler
M0.sample(iter=50000, burn=10000, thin=5, progress_bar=True)

# Get the mean results and the 95% confidence level
mean = M0.stats()['fourier_power_spectrum']['mean']
low = M0.stats()['fourier_power_spectrum']['95% HPD interval'][:, 0]
high = M0.stats()['fourier_power_spectrum']['95% HPD interval'][:, 1]


