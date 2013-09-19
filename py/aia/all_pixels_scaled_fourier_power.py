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


if False:
    location = '~/Data/oscillations/mcateer/outgoing3/AR_A.sav'
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
    wave = '171'

    # Construct the directory
    directory = os.path.join(maindir, wave)

    # Load in the data
    print('Loading data from ' + directory)
    dc = get_datacube(directory)


#wave = '171'
#dc, location = aia_specific.rn4(wave, Xrange=[-201, -1])

# Get some properties of the datacube
ny = dc.shape[0]
nx = dc.shape[1]
nt = dc.shape[2]

# Create a time series object
dt = 12.0
t = dt * np.arange(0, nt)
tsdummy = TimeSeries(t, t)
iobs = np.zeros(tsdummy.PowerSpectrum.Npower.shape)
logiobs = np.zeros(tsdummy.PowerSpectrum.Npower.shape)
nposfreq = len(iobs)

# Result # 1 - add up all the emission and do the analysis on the full FOV
# Also, make a histogram of all the power spectra to get an idea of the
# varition present

# storage
pwr = np.zeros((ny, nx, nposfreq))
logpwr = np.zeros_like(pwr)
full_ts = np.zeros((nt))
for i in range(0, nx):
    for j in range(0, ny):
        d = dc[j, i, :].flatten()
        # Fix the data for any non-finite entries
        d = tsutils.fix_nonfinite(d)

        # Remove the mean
        d = d - np.mean(d)
        # Express in units of the standard deviation of the time series
        #d = d / np.std(d)

        # Form a time series object.  Handy for calculating the Fourier power
        # at all non-zero frequencies
        ts = TimeSeries(t, d)

        # Define the Fourier power we are analyzing
        this_power = ts.PowerSpectrum.ppower

        # Look at the power
        iobs = iobs + this_power
        pwr[j, i, :] = this_power

        # Look at the log of the power
        logiobs = logiobs + np.log(this_power)
        logpwr[j, i, :] = np.log(this_power)


# Average power over all the pixels
iobs = iobs / (1.0 * nx * ny)

# Express the power in each frequency as a multiple of the average
av_iobs = np.mean(iobs)
iobs = iobs / av_iobs

# Express the power in each frequency as a multiple of the average for all
# Fourier power at each pixel
pwr = pwr / av_iobs

# Normalize the frequency.
freqs = tsdummy.PowerSpectrum.frequencies.positive
x = freqs / tsdummy.PowerSpectrum.frequencies.positive[0]

# Sigma for the fit to the power
sigma = np.std(pwr, axis=(0, 1))


# function we are fitting
def func(freq, a, n, c):
    return a * freq ** -n + c

# do the fit
answer = curve_fit(func, x, iobs, sigma=sigma)

# Get the fit parameters out and calculate the best fit
param = answer[0]
bf = func(x, param[0], param[1], param[2])

# Error estimate for the power law index
nerr = np.sqrt(answer[1][1, 1])

# Give the best plot we can under the circumstances.
plt.figure()
plt.loglog(freqs, iobs, label='arithmetic mean')
plt.loglog(freqs, bf, color='k', label='best fit n=%4.2f +/- %4.2f' % (param[1], nerr))
plt.axvline(1.0 / 300.0, color='k', linestyle='-.', label='5 mins.')
plt.axvline(1.0 / 180.0, color='k', linestyle='--', label='3 mins.')
plt.xlabel('frequency (Hz)')
plt.ylabel('normalized power [%i time series, %i samples each]' % (nx * ny, nt))
plt.title('AIA ' + str(wave) + ': ' + location)
plt.legend(loc=3, fontsize=10)
plt.ylim(0.0001, 1000.0)
plt.show()

# ------------------------------------------------------------------------
# Do the same thing over again, this time working with the log of the
# normalized power
# Average power over all the pixels
logiobs = logiobs / (1.0 * nx * ny)

# Sigma for the log of the power over all pixels
logsigma = np.std(logpwr, axis=(0, 1))


# Log of the power spectrum model
def func2(freq, a, n, c):
    return np.log(func(freq, a, n, c))

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
plt.loglog(freqs, lim[0, 0, :], linestyle='--', label='lower 68%')
plt.loglog(freqs, lim[0, 1, :], linestyle='--', label='upper 68%')
plt.loglog(freqs, lim[1, 0, :], linestyle=':', label='lower 95%')
plt.loglog(freqs, lim[1, 1, :], linestyle=':', label='upper 95%')
plt.loglog(freqs, bf2, color='k', label='best fit n=%4.2f +/- %4.2f' % (param2[1], nerr2))
plt.axvline(1.0 / 300.0, color='k', linestyle='-.', label='5 mins.')
plt.axvline(1.0 / 180.0, color='k', linestyle='--', label='3 mins.')
plt.xlabel('frequency (Hz)')
plt.ylabel('normalized power [%i time series, %i samples each]' % (nx * ny, nt))
plt.title('AIA ' + str(wave) + ': ' + location)
plt.legend(loc=3, fontsize=10)
plt.show()

# plot some histograms of the log power at a small number of equally spaced
# frequencies
findex = np.arange(0, nposfreq, nposfreq / 5)
plt.figure()
plt.xlabel('$\log_{10}(power)$')
plt.ylabel('proportion found at given frequency')
plt.title('AIA ' + str(wave) + ': ' + location)
for f in findex:
    plt.plot(h[1][1:] / np.log(10.0), hpwr[f, :], label='%7.5f Hz' % (freqs[f]))
plt.legend(loc=3, fontsize=10)
plt.show()
