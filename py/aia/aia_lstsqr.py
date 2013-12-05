"""
Specifies a directory of AIA files and uses a least-squares fit to generate
some fit estimates.
"""
# Test 6: Posterior predictive checking
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import tsutils
import os
from timeseries import TimeSeries
from scipy.optimize import curve_fit
from rnsimulation import SimplePowerLawSpectrumWithConstantBackground, TimeSeriesFromPowerSpectrum
from rnfit2 import Do_MCMC, rnsave
from pymcmodels import single_power_law_with_constant_not_normalized
from cubetools import get_datacube
import scipy.stats as stats
import cPickle as pickle


font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)
matplotlib.rc('lines', linewidth=2)
matplotlib.rc('figure', figsize=(12.5, 10))

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

# Load in the datacube
directory = '~/ts/pickle/shutdownfun3'
wave = '211'
region = 'loopfootpoints'
savefig = '~/ts/img/shutdownfun3_1hr'

location = os.path.join(os.path.expanduser(directory), wave)
filename = region + '.' + wave + '.datacube.pickle'
pkl_file_location = os.path.join(location, filename)
print('Loading ' + pkl_file_location)
pkl_file = open(pkl_file_location, 'rb')
dc = pickle.load(pkl_file)
pkl_file.close()

# Create a location to save the figures
savefig = os.path.join(os.path.expanduser(savefig), wave)
if not(os.path.isdir(savefig)):
    os.makedirs(savefig)
figname = wave + '.' + region
savefig = os.path.join(savefig, figname)



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
full_data = np.zeros((nt))
for i in range(0, nx):
    for j in range(0, ny):
        d = dc[j, i, :].flatten()
        # Fix the data for any non-finite entries
        d = tsutils.fix_nonfinite(d)

        # Sum up all the data
        full_data = full_data + d

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

# Average emission over all the data
full_data = full_data / (1.0 * nx * ny)

# Create a time series object
full_ts = TimeSeries(t, full_data)
full_ts.name = 'AIA '+wave+': '+region 
full_ts.label = 'average emission [%i time series]' % (nx * ny)
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

# Log of the power spectrum model
def func2(freq, a, n, c):
    print a, n, c
    z = func(freq, a, n, c)
    print z
    return np.log(z)


# do the fit
answer = curve_fit(func, x, iobs, sigma=sigma)

# Get the fit parameters out and calculate the best fit
param = answer[0]
bf = func(x, param[0], param[1], param[2])

# Error estimate for the power law index
nerr = np.sqrt(answer[1][1, 1])

# Analyze the summed time-series
# do the fit
full_ts_iobs = full_ts.PowerSpectrum.ppower / np.mean(full_ts.PowerSpectrum.ppower)
answer_full_ts = curve_fit(func2, x, np.log(full_ts_iobs), p0=answer[0])
# Get the fit parameters out and calculate the best fit
param_fts = answer_full_ts[0]
bf_fts = np.exp(func2(x, param_fts[0], param_fts[1], param_fts[2]))
nerr_fts = np.sqrt(answer[1][1, 1])

# Give the best plot we can under the circumstances.
plt.figure(1)
plt.loglog(freqs, iobs, label='arithmetic mean of power spectra from each pixel (Erlang distributed)', color='b')
plt.loglog(freqs, bf, color='b', linestyle="--", label='fit to arithmetic mean of power spectra from each pixel n=%4.2f +/- %4.2f' % (param[1], nerr))
plt.loglog(freqs, full_ts_iobs, label='power spectrum from summed emission (exponential distributed)', color='r')
plt.loglog(freqs, bf_fts, label='fit to power spectrum of summed emission n=%4.2f +/- %4.2f' % (param_fts[1], nerr_fts), color='r', linestyle="--")
plt.axvline(1.0 / 300.0, color='k', linestyle='-.', label='5 mins.')
plt.axvline(1.0 / 180.0, color='k', linestyle='--', label='3 mins.')
plt.axhline(1.0, color='k', label='average power')
plt.xlabel('frequency (Hz)')
plt.ylabel('normalized power [%i time series, %i samples each]' % (nx * ny, nt))
plt.title('AIA ' + str(wave) + ': ' + region)
plt.legend(loc=3, fontsize=10)
plt.text(freqs[0], 500, 'note: least-squares fit used, but data is not Gaussian distributed', fontsize=8)
plt.ylim(0.0001, 1000.0)
plt.savefig(savefig + '.arithmetic_mean_power_spectra.png')

# ------------------------------------------------------------------------
# Do the same thing over again, this time working with the log of the
# normalized power
# Average power over all the pixels
logiobs = logiobs / (1.0 * nx * ny)

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

# number of histogram bins
bins = 100
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

plt.figure(2)
plt.loglog(freqs, np.exp(logiobs), label='geometric mean of power spectra at each pixel')
plt.loglog(freqs, bf2, color='k', label='best fit n=%4.2f +/- %4.2f' % (param2[1], nerr2))
plt.loglog(freqs, lim[0, 0, :], linestyle='--', label='lower 68%')
plt.loglog(freqs, lim[0, 1, :], linestyle='--', label='upper 68%')
plt.loglog(freqs, lim[1, 0, :], linestyle=':', label='lower 95%')
plt.loglog(freqs, lim[1, 1, :], linestyle=':', label='upper 95%')
plt.axvline(1.0 / 300.0, color='k', linestyle='-.', label='5 mins.')
plt.axvline(1.0 / 180.0, color='k', linestyle='--', label='3 mins.')
plt.xlabel('frequency (Hz)')
plt.ylabel('power [%i time series, %i samples each]' % (nx * ny, nt))
plt.title('AIA ' + str(wave) + ': ' + region)
plt.legend(loc=3, fontsize=10)
plt.savefig(savefig + '.geometric_mean_power_spectra.png')

# plot some histograms of the log power at a small number of equally spaced
# frequencies
findex = np.arange(0, nposfreq, nposfreq / 5)
plt.figure(3)
plt.xlabel('$\log_{10}(power)$')
plt.ylabel('proportion found at given frequency')
plt.title('AIA ' + str(wave) + ': ' + region)
for f in findex:
    plt.plot(h[1][1:] / np.log(10.0), hpwr[f, :], label='%7.5f Hz' % (freqs[f]))
plt.legend(loc=3, fontsize=10)
plt.savefig(savefig + '.power_spectra_distributions.png')


# plot out the time series
plt.figure(4)
full_ts.peek()
plt.savefig(savefig + '.full_ts_timeseries.png')



#
# Make maps of the Fourier power
#
fmap = []
franges = [[1.0/360.0, 1.0/240.0], [1.0/240.0, 1.0/120.0]]
for fr in franges:
    ind = []
    for i, testf in enumerate(freqs):
        if testf >= fr[0] and testf <= fr[1]:
            ind.append(i)
    fmap.append(np.sum(pwr[:,:,ind[:]], axis=2))

#
# Do the MCMC stuff
#
# Normalize the frequency so that the first element is equal to 1 and
# all higher frequencies are multiples of the first non-zero frequency.  This
# makes calculation slightly easier.

# Form the input for the MCMC algorithm.
this = ([freqs, full_ts_iobs],)


norm_estimate = np.zeros((3,))
norm_estimate[0] = iobs[0]
norm_estimate[1] = norm_estimate[0] / 1000.0
norm_estimate[2] = norm_estimate[0] * 1000.0

background_estimate = np.zeros_like(norm_estimate)
background_estimate[0] = np.mean(iobs[-10:-1])
background_estimate[1] = background_estimate[0] / 1000.0
background_estimate[2] = background_estimate[0] * 1000.0

estimate = {"norm_estimate": norm_estimate,
            "background_estimate": background_estimate}

# _____________________________________________________________________________
# -----------------------------------------------------------------------------
# Analyze using MCMC
# -----------------------------------------------------------------------------
analysis = Do_MCMC(this).okgo(single_power_law_with_constant_not_normalized,
                              estimate=estimate,
                              seed=None,
                              iter=50000,
                              burn=10000,
                              thin=5,
                              progress_bar=True)

analysis.showfit(figure=5, show_simulated=[0], show_1=True,
                 title='AIA '+ wave + ' ' + region + ': '+ 'Bayesian/MCMC fit')
plt.savefig(savefig + '.mcmc_fit_with_stochastic_estimate.png')
