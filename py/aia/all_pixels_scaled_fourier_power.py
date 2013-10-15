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
#from rnfit2 import Do_MCMC, rnsave
#import ppcheck2
#from pymcmodels import single_power_law_with_constant
import os
from timeseries import TimeSeries
from scipy.optimize import curve_fit
from scipy.io import readsav
from cubetools import get_datacube
#from rnsimulation import SimplePowerLawSpectrumWithConstantBackground, TimeSeriesFromPowerSpectrum
import aia_specific
plt.ion()

"""
For any input datacube we wish to define a certain number of data products...
(1) full_data : sum up all the emission in a region to get a single time series

(2) Fourier power : for each pixel in the region calculate the Fourier power

(3) Average Fourier power : the arithmetic mean of the Fourier power (2)

(4) Log Fourier power : the log of the Fourier power for each pixel in the
                        region (2).

(5) Average log Fourier power : the average of the log Fourier power (4).  This
                                is the same as the geometric mean of the
                                Fourier power (2).

(6) A set of pixel locations that can be shared across datasets that image
    different wavebands.

This is what we do with the data and how we do it:

(a) Fit (1) using the MCMC algorithm.

(b) Fit (3) using a least-squares fitting algorithm (yields an approximate
    answer which will be similar to that of (a))

(c)
"""


if False:
    location = '~/Data/oscillations/mcateer/outgoing3/AR_A.sav'
    maindir = os.path.expanduser(location)
    print('Loading data from ' + maindir)
    wave = '???'
    idl = readsav(maindir)
    dc = np.swapaxes(np.swapaxes(idl['region_window'], 0, 2), 0, 1)
else:
    wave = '171'
    AR = False
    if AR:
        Xrange = [0, 300]
        Yrange = [200, 350]
        type = 'AR'
    else:
        Xrange = [550, 650]
        Yrange = [20, 300]
        type = 'QS'
    dc, location = aia_specific.rn4(wave,
                                    '~/Data/AIA/shutdownfun2/',
                                    Xrange=Xrange,
                                    Yrange=Yrange)



# Get some properties of the datacube
ny = dc.shape[0]
nx = dc.shape[1]
nt = dc.shape[2]

# Create a time series object
dt = 12.0
t = dt * np.arange(0, nt)

# Fix the input datacube for any non-finite data
for i in range(0, nx):
    for j in range(0, ny):
        d = dc[j, i, :].flatten()
        # Fix the data for any non-finite entries
        d = tsutils.fix_nonfinite(d)
        # Remove the mean
        d = d - np.mean(d)
        dc[j, i, :] = d

# Time series datacube
dcts = TimeSeries(t, dc)
dcts.name = 'AIA ' + str(wave) + '-' + type + ': ' + location

# Get the Fourier power
pwr = dcts.ppower

# Arithmetic mean of the Fourier power
iobs = np.mean(pwr, axis=(0, 1))

# Sigma for the fit to the power
sigma = np.std(pwr, axis=(0, 1))

# Result # 1 - add up all the emission and do the analysis on the full FOV
# Also, make a histogram of all the power spectra to get an idea of the
# varition present

# Create a time series object of the average emission over the whole datacube
full_ts = TimeSeries(t, np.mean(dcts.data, axis=(0, 1)))

# Normalize the frequency.
freqs = dcts.PowerSpectrum.frequencies.positive
x = freqs / dcts.PowerSpectrum.frequencies.positive[0]


# function we are fitting
def func(freq, a, n, c):
    return a * freq ** -n + c


# Log of the power spectrum model
def func2(freq, a, n, c):
    return np.log(func(freq, a, n, c))


# do the fit to the arithmetic mean
answer = curve_fit(func, x, iobs, sigma=sigma, p0=[iobs[0], 2, iobs[-1]])

pwrlaw = np.zeros((ny, nx))
norm = np.zeros((ny, nx))
for i in range(0, ny-1):
    for j in range(0, nx-1):
        y = pwr[i, j, :]
        try:
            aaa = curve_fit(func, x, y, sigma=sigma, p0=[y[0], 2, y[-1]])
            pwrlaw[i, j] = aaa[0][1]
            norm[i, j] = aaa[0][0]
        except:
            pwrlaw[i,j] = -1
            norm[i, j] = -1

# Get the fit parameters out and calculate the best fit spectrum
param = answer[0]
bf = func(x, param[0], param[1], param[2])

# Error estimate for the power law index
nerr = np.sqrt(answer[1][1, 1])

# Analyze the summed time-series
# do the fit
full_ts_iobs = full_ts.PowerSpectrum.ppower
answer_full_ts = curve_fit(func2, x, np.log(full_ts_iobs), p0=answer[0])
# Get the fit parameters out and calculate the best fit
param_fts = answer_full_ts[0]
bf_fts = np.exp(func2(x, param_fts[0], param_fts[1], param_fts[2]))
nerr_fts = np.sqrt(answer[1][1, 1])

# Plot the 
plt.figure()
plt.loglog(freqs, iobs, label='arithmetic mean of the power spectra in every pixel (Erlang distributed)', color='b')
plt.loglog(freqs, bf, color='b', linestyle="--", label='fit to arithmetic mean n=%4.2f +/- %4.2f' % (param[1], nerr))
plt.loglog(freqs, full_ts_iobs, label='summed emission (exponential distributed)', color='r')
plt.loglog(freqs, bf_fts, label='fit to summed emission n=%4.2f +/- %4.2f' % (param_fts[1], nerr_fts), color='r', linestyle="--")
plt.axvline(1.0 / 300.0, color='k', linestyle='-.', label='5 mins.')
plt.axvline(1.0 / 180.0, color='k', linestyle='--', label='3 mins.')
plt.xlabel('frequency (Hz)')
plt.ylabel('normalized power [%i time series, %i samples each]' % (nx * ny, nt))
plt.title(dcts.name)
plt.legend(loc=3, fontsize=10)
plt.text(freqs[0], 500, 'note: least-squares fit used, but data is not Gaussian distributed', fontsize=8)
plt.ylim(0.0001, 1000.0)
plt.show()

# ------------------------------------------------------------------------
# Do the same thing over again, this time working with the log of the
# normalized power

# Get the logarithm of the Fourier power
logpwr = np.log(pwr)

# Geometric mean of the Fourier power
logiobs = np.mean(logpwr, axis=(0,1))

# Sigma for the log of the power over all pixels
logsigma = np.std(logpwr, axis=(0, 1))

# Fit the function to the log of the mean power
answer2 = curve_fit(func2, x, logiobs, sigma=logsigma, p0=answer[0])

# Get the fit parameters out and calculate the best fit
param2 = answer2[0]
bf2 = np.exp(func2(x, param2[0], param2[1], param2[2]))

# Error estimate for the power law index
nerr2 = np.sqrt(answer2[1][1, 1])

# Create the histogram of all the log powers.  Histograms look normal-ish if
# you take the logarithm of the power.  This suggests a log-normal distribution
# of power in all frequencies

nposfreq = len(x)
# number of histogram bins
bins = 1000
hpwr = np.zeros((nposfreq, bins))
for f in range(0, nposfreq):
    h = np.histogram(logpwr[:, :, f], bins=bins, range=[np.min(logpwr), np.max(logpwr)])
    hpwr[f, :] = h[0] / (1.0 * np.sum(h[0]))

# Calculate the probability density in each frequency bin.
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

# Give the best plot we can under the circumstances.  Since we have been
# looking at the log of the power, plots are slightly different

plt.figure()
plt.loglog(freqs, np.exp(logiobs), label='geometric mean')
plt.loglog(freqs, bf2, color='k', label='best fit n=%4.2f +/- %4.2f' % (param2[1], nerr2))
plt.loglog(freqs, lim[0, 0, :], linestyle='--', label='lower 68%')
plt.loglog(freqs, lim[0, 1, :], linestyle='--', label='upper 68%')
plt.loglog(freqs, lim[1, 0, :], linestyle=':', label='lower 95%')
plt.loglog(freqs, lim[1, 1, :], linestyle=':', label='upper 95%')
plt.axvline(1.0 / 300.0, color='k', linestyle='-.', label='5 mins.')
plt.axvline(1.0 / 180.0, color='k', linestyle='--', label='3 mins.')
plt.xlabel('frequency (Hz)')
plt.ylabel('normalized power [%i time series, %i samples each]' % (nx * ny, nt))
plt.title(dcts.name)
plt.legend(loc=3, fontsize=10)
plt.show()

# plot some histograms of the log power at a small number of equally spaced
# frequencies
findex = np.arange(0, nposfreq, nposfreq / 5)
plt.figure()
plt.xlabel('$\log_{10}(power)$')
plt.ylabel('proportion found at given frequency')
plt.title(dcts.name)
for f in findex:
    plt.plot(h[1][1:] / np.log(10.0), hpwr[f, :], label='%7.5f Hz' % (freqs[f]))
plt.legend(loc=3, fontsize=10)
plt.show()

