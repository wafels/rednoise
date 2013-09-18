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
#from rnfit2 import Do_MCMC, rnsave
#import ppcheck2
#from pymcmodels import single_power_law_with_constant
from cubetools import get_datacube
from timeseries import TimeSeries
from scipy.io import readsav
from scipy.optimize import curve_fit
#from rnsimulation import SimplePowerLawSpectrumWithConstantBackground, TimeSeriesFromPowerSpectrum


if True:
    location = '~/Data/oscillations/mcateer/outgoing3/CH_I.sav'
    maindir = os.path.expanduser(location)
    print('Loading data from ' + maindir)
    wave = '???'
    idl = readsav(maindir)
    dc = np.swapaxes(np.swapaxes(idl['region_window'], 0, 2), 0, 1)
else:
    # Main directory where the data is
    location = '~/Data/AIA_Data/SOL2011-04-30T21-45-49L061C108'
    maindir = os.path.expanduser(location)

    # Which wavelength to look at
    wave = '193'

    # Construct the directory
    directory = os.path.join(maindir, wave)

    # Load in the data
    print('Loading data from ' + directory)
    dc = get_datacube(directory)

# Get some properties of the datacube
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
bins = 1000

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

# Fit a simple model to the data
x = tsdummy.PowerSpectrum.frequencies.positive / tsdummy.PowerSpectrum.frequencies.positive[0]

# Sigma for the fit
sigma = np.std(np.exp(pwr), axis=(0, 1))

def func(x, a, n, c):
    return a * x ** -n + c

answer = curve_fit(func, x, iobs, sigma=sigma)

param = answer[0]
bf = func(x, param[0], param[1], param[2])

# Create the histogram of all the powers
hpwr = np.zeros((nposfreq, bins))
for f in range(0, nposfreq):
    h = np.histogram(pwr[:, :, f], bins=bins, range=[pwr.min(), pwr.max()])
    hpwr[f, :] = h[0] / (1.0 * np.sum(h[0]))

p = [0.68, 0.95]
lim = np.zeros((len(p), 2, nposfreq))
for i, thisp in enumerate(p):
    tailp = 0.5 * (1.0 - thisp)
    for f in range(0, nposfreq):
        lo = 0
        while np.sum(hpwr[f, 0:lo]) <= tailp:
            lo = lo + 1
        hi = 0
        while np.sum(hpwr[f, 0:hi]) <= 1.0 - tailp:
            hi = hi + 1
        lim[i, 0, f] = np.exp(h[1][lo])
        lim[i, 1, f] = np.exp(h[1][hi])

plt.figure()
plt.loglog(tsdummy.PowerSpectrum.frequencies.positive, iobs, label='mean')
plt.loglog(tsdummy.PowerSpectrum.frequencies.positive, lim[0, 0, :], linestyle='--', label='lower 68%')
plt.loglog(tsdummy.PowerSpectrum.frequencies.positive, lim[0, 1, :], linestyle='--', label='upper 68%')
plt.loglog(tsdummy.PowerSpectrum.frequencies.positive, lim[1, 0, :], linestyle=':', label='lower 95%')
plt.loglog(tsdummy.PowerSpectrum.frequencies.positive, lim[1, 1, :], linestyle=':', label='upper 95%')
plt.loglog(tsdummy.PowerSpectrum.frequencies.positive, bf, color='k', label='best fit n=%4.2f' %(param[1]))
plt.axvline(1.0 / 300.0, color='k', linestyle='-.', label='5 mins.')
plt.axvline(1.0 / 180.0, color='k', linestyle='--', label='3 mins.')
plt.xlabel('frequency (Hz)')
plt.ylabel('normalized power [%i time series, %i samples each]' % (nx * ny, nt))
plt.title('AIA ' + str(wave) + ': ' + location)
plt.legend(loc=3)
plt.ylim(0.0001, 100.0)
plt.show()
