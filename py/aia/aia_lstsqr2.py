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
from scipy.optimize import curve_fit, fmin_tnc
from rnsimulation import SimplePowerLawSpectrumWithConstantBackground, TimeSeriesFromPowerSpectrum
from rnfit2 import Do_MCMC, rnsave
from pymcmodels import single_power_law_with_constant_not_normalized
from cubetools import get_datacube
import scipy.stats as stats
import csv
import cPickle as pickle


font = {'family': 'normal',
        'weight': 'bold',
        'size': 22}

matplotlib.rc('font', **font)
matplotlib.rc('lines', linewidth=2)
matplotlib.rc('figure', figsize=(12.5, 10))

plt.ioff()


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


# function we are fitting
def PowerLawPlusConstant(freq, a, n, c):
    return a * a * freq ** -n + c


# Log of the power spectrum model
def LogPowerLawPlusConstant(freq, a, n, c):
    z = PowerLawPlusConstant(freq, a, n, c)
    return np.log(z)


# String defining the basics number of time series
def tsDetails(nx, ny, nt):
    return '[%i t.s., %i samples]' % (nx * ny, nt)


# Apodization windowing function
def DefineWindow(window, nt):
    if window == 'no window':
        win = np.ones(nt)
    if window == 'hanning':
        win = np.hanning(nt)
    if window == 'hamming':
        win = np.hamming(nt)
    winname = ', ' + window
    return win, winname


# Main analysis loop
def do_lstsqr(dataroot='~/Data/AIA/',
              ldirroot='~/ts/pickle/',
              sfigroot='~/ts/img/',
              scsvroot='~/ts/csv/',
              corename='shutdownfun3_6hr',
              sunlocation='disk',
              fits_level='1.5',
              waves=['171', '193', '211', '131'],
              regions=['qs', 'loopfootpoints'],
              windows=['no window'],
              manip='none'):

    ldir = os.path.join(os.path.expanduser(ldirroot), corename)
    sfig = os.path.join(os.path.expanduser(sfigroot), corename)
    scsv = os.path.join(os.path.expanduser(scsvroot), corename)

    frequency_factor = 1000.0
    five_min = 1.0 / 300.0
    three_min = 1.0 / 180.0


    # main loop
    for iwave, wave in enumerate(waves):
        print('Wave: ' + wave + ' (%i out of %i)' % (iwave + 1, len(waves)))
        # Identifier
        ident = corename + '_' + sunlocation + '_' + fits_level + '_' + wave
        for iregion, region in enumerate(regions):
            print('Region: ' + region + ' (%i out of %i)' % (iregion + 1, len(regions)))
            region_id = region + '.' + ident
            for iwindow, window in enumerate(windows):
                print('Window: ' + window + ' (%i out of %i)' % (iwindow + 1, len(windows)))

                # set the loading and saving locations
                directory = ldir
                savefig = sfig
                savecsv = scsv

                # load the data
                location = os.path.join(os.path.expanduser(directory), sunlocation, fits_level, wave)
                filename = region_id + '.datacube.pickle'
                pkl_file_location = os.path.join(location, filename)
                print('Loading ' + pkl_file_location)
                pkl_file = open(pkl_file_location, 'rb')
                dc = pickle.load(pkl_file)
                pkl_file.close()

                # Get some properties of the datacube
                ny = dc.shape[0]
                nx = dc.shape[1]
                nt = dc.shape[2]
                tsdetails = tsDetails(nx, ny, nt)

                # Define an array to store the analyzed data
                dc_analysed = np.zeros_like(dc)

                # calculate a window function
                win, winname = DefineWindow(window, nt)

                # Create the name for the data
                data_name = wave + ' (' + fits_level + winname + ', ' + manip + '), ' + region

                # Create a location to save the figures
                savefig = os.path.join(os.path.expanduser(savefig), fits_level, wave, region, window, manip)
                if not(os.path.isdir(savefig)):
                    os.makedirs(savefig)
                figname = data_name
                savefig = os.path.join(savefig, figname)

                # Create a time series object
                dt = 12.0
                t = dt * np.arange(0, nt)
                tsdummy = TimeSeries(t, t)
                freqs = tsdummy.PowerSpectrum.frequencies.positive
                iobs = np.zeros(tsdummy.PowerSpectrum.Npower.shape)
                logiobs = np.zeros(tsdummy.PowerSpectrum.Npower.shape)
                nposfreq = len(iobs)

                # storage
                pwr = np.zeros((ny, nx, nposfreq))
                logpwr = np.zeros_like(pwr)

                for i in range(0, nx):
                    for j in range(0, ny):

                        # Get the next time-series
                        d = dc[j, i, :].flatten()

                        # Fix the data for any non-finite entries
                        d = tsutils.fix_nonfinite(d)

                        # Remove the mean
                        #if manip == 'submean':
                        #    d = d - np.mean(d)

                        # Relative intensity
                        if manip == 'relative':
                            dmean = np.mean(d)
                            d = (d - dmean) / dmean

                        # Multiply the data by the apodization window
                        d = d * win

                        # Keep the analyzed data cube
                        dc_analysed[j, i, :] = d

                        # Form a time series object.  Handy for calculating the Fourier power at all non-zero frequencies
                        ts = TimeSeries(t, d)

                        # Define the Fourier power we are analyzing
                        this_power = ts.PowerSpectrum.ppower

                        # Get the total Fourier power
                        iobs = iobs + this_power

                        # Store the individual Fourier power
                        pwr[j, i, :] = this_power

                        # Sum up the log of the Fourier power - equivalent to doing the geometric mean
                        logiobs = logiobs + np.log(this_power)

                        # Store the individual log Fourier power
                        logpwr[j, i, :] = np.log(this_power)

                #
                dc_analysed_minmax = (np.min(dc_analysed), np.max(dc_analysed))

                # Plot all the analyzed time series
                plt.figure(10)
                for i in range(0, nx):
                    for j in range(0, ny):
                        plt.plot(t, dc_analysed[j, i, :])
                plt.xlabel('time (seconds)')
                plt.ylabel('analyzed emission ' + tsdetails)
                plt.title(data_name)
                plt.ylim(dc_analysed_minmax)
                plt.xlim((t[0], t[-1]))
                plt.savefig(savefig + '.all_analyzed_ts.png')

                # Plot a histogram of the studied data at each time
                bins = 50
                hist_dc_analysed = np.zeros((bins, nt))
                for this_time in range(0, nt):
                    hist_dc_analysed[:, this_time], bin_edges = np.histogram(dc_analysed[:, :, this_time], bins=bins, range=dc_analysed_minmax)
                hist_dc_analysed = hist_dc_analysed / (1.0 * nx * ny)
                plt.figure(12)
                plt.xlabel('time (seconds)')
                plt.ylabel('analyzed emission ' + tsdetails)
                plt.imshow(hist_dc_analysed, aspect='auto', origin='lower',
                           extent=(t[0], t[-1], dc_analysed_minmax[0], dc_analysed_minmax[1]))
                plt.colorbar()
                plt.title(data_name)
                plt.savefig(savefig + '.all_analyzed_ts_histogram.png')

                # Plot all the analyzed FFTs
                plt.figure(11)
                for i in range(0, nx):
                    for j in range(0, ny):
                        ts = TimeSeries(t, dc_analysed[j, i, :])
                        ts.peek_ps()
                plt.loglog()
                plt.axvline(five_min, color='k', linestyle='-.', label='5 mins.')
                plt.axvline(three_min, color='k', linestyle='--', label='3 mins.')
                plt.xlabel('frequency (Hz)')
                plt.ylabel('FFT power ' + tsdetails)
                plt.title(data_name)
                plt.savefig(savefig + '.all_analyzed_fft.png')

                # Plot a histogram of the studied FFTs at each time
                bins = 50
                minmax = [np.min(logpwr), np.max(logpwr)]
                hist_dc_analysed_logpwr = np.zeros((bins, nposfreq))
                for this_freq in range(0, nposfreq):
                    hist_dc_analysed_logpwr[:, this_freq], bin_edges = np.histogram(logpwr[:, :, this_freq], bins=bins, range=minmax)
                hist_dc_analysed_logpwr = hist_dc_analysed_logpwr / (1.0 * nx * ny)
                plt.figure(13)
                plt.xlabel('frequency (Hz)')
                plt.ylabel('FFT power ' + tsdetails)
                plt.imshow(hist_dc_analysed_logpwr, aspect='auto', origin='lower',
                           extent=(freqs[0], freqs[-1], np.exp(minmax[0]), np.exp(minmax[1])))
                plt.semilogy()
                plt.colorbar()
                plt.title(data_name)
                plt.savefig(savefig + '.all_analyzed_fft_histogram.png')


                # Plot all the Fourier powers
                # Average of the analyzed time-series and create a time series object
                full_data = np.mean(dc_analysed, axis=(0, 1))
                full_ts = TimeSeries(t, full_data)
                full_ts.name = data_name
                full_ts.label = 'average emission ' + tsdetails
                # Average power over all the pixels
                iobs = iobs / (1.0 * nx * ny)

                # Save the full time series to a CSV file
                savecsv = os.path.join(os.path.expanduser(savecsv), fits_level, wave, region, window, manip)
                if not(os.path.isdir(savecsv)):
                    os.makedirs(savecsv)
                savecsv = os.path.join(savecsv, data_name)
                ofile = open(savecsv + '.full_ts.csv', "wb")
                writer = csv.writer(ofile, delimiter=',')
                for i, d in enumerate(full_data):
                    writer.writerow([t[i], d])
                ofile.close()

                # Express the power in each frequency as a multiple of the average
                #av_iobs = np.mean(iobs)
                #iobs = iobs / av_iobs

                # Express the power in each frequency as a multiple of the average for all
                # Fourier power at each pixel
                #pwr = pwr / av_iobs

                # Normalize the frequency.
                x = freqs / tsdummy.PowerSpectrum.frequencies.positive[0]

                # Sigma for the fit to the power
                sigma = np.std(pwr, axis=(0, 1))

                # Generate a guess
                pguess = [iobs[0], 1.8, iobs[-1]]

                # do the fit
                answer = curve_fit(PowerLawPlusConstant, x, iobs, sigma=sigma, p0=pguess)

                # Get the fit parameters out and calculate the best fit
                param = answer[0]
                bf = PowerLawPlusConstant(x, param[0], param[1], param[2])

                # Error estimate for the power law index
                nerr = np.sqrt(answer[1][1, 1])

                # Analyze the summed time-series
                # do the fit
                full_ts_iobs = full_ts.PowerSpectrum.ppower / np.mean(full_ts.PowerSpectrum.ppower)
                answer_full_ts = curve_fit(LogPowerLawPlusConstant, x, np.log(full_ts_iobs), p0=answer[0])
                #answer_full_ts = fmin_tnc(LogPowerLawPlusConstant, x, approx_grad=True)
                # Get the fit parameters out and calculate the best fit
                param_fts = answer_full_ts[0]
                bf_fts = np.exp(LogPowerLawPlusConstant(x, param_fts[0], param_fts[1], param_fts[2]))
                nerr_fts = np.sqrt(answer[1][1, 1])

                # Give the best plot we can under the circumstances.
                plt.figure(1)
                plt.loglog(freqs, iobs, label='arithmetic mean of power spectra from each pixel (Erlang distributed)', color='b')
                plt.loglog(freqs, bf, color='b', linestyle="--", label='fit to arithmetic mean of power spectra from each pixel n=%4.2f +/- %4.2f' % (param[1], nerr))
                plt.loglog(freqs, full_ts_iobs, label='power spectrum from summed emission (exponential distributed)', color='r')
                plt.loglog(freqs, bf_fts, label='fit to power spectrum of summed emission n=%4.2f +/- %4.2f' % (param_fts[1], nerr_fts), color='r', linestyle="--")
                plt.axvline(five_min, color='k', linestyle='-.', label='5 mins.')
                plt.axvline(three_min, color='k', linestyle='--', label='3 mins.')
                plt.axhline(1.0, color='k', label='average power')
                plt.xlabel('frequency (Hz)')
                plt.ylabel('normalized power [%i time series, %i samples each]' % (nx * ny, nt))
                plt.title(data_name + ' - aPS')
                plt.legend(loc=1, fontsize=10)
                plt.text(freqs[0], 500, 'note: least-squares fit used, but data is not Gaussian distributed', fontsize=8)
                #plt.ylim(0.0001, 1000.0)
                plt.savefig(savefig + '.arithmetic_mean_power_spectra.png')

                # ------------------------------------------------------------------------
                # Do the same thing over again, this time working with the log of the
                # normalized power
                # Average power over all the pixels
                logiobs = logiobs / (1.0 * nx * ny)

                # Sigma for the log of the power over all pixels
                logsigma = np.std(logpwr, axis=(0, 1))

                # Fit the function to the log of the mean power
                answer2 = curve_fit(LogPowerLawPlusConstant, x, logiobs, sigma=logsigma, p0=answer[0])

                # Get the fit parameters out and calculate the best fit
                param2 = answer2[0]
                bf2 = np.exp(LogPowerLawPlusConstant(x, param2[0], param2[1], param2[2]))

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
                plt.axvline(five_min, color='k', linestyle='-.', label='5 mins.')
                plt.axvline(three_min, color='k', linestyle='--', label='3 mins.')
                plt.xlabel('frequency (Hz)')
                plt.ylabel('power [%i time series, %i samples each]' % (nx * ny, nt))
                plt.title(data_name + ' - gPS')
                plt.legend(loc=1, fontsize=10)
                plt.savefig(savefig + '.geometric_mean_power_spectra.png')

                # plot some histograms of the log power at a small number of equally spaced
                # frequencies
                findex = [0, 11, 19, 38, 76]
                plt.figure(3)
                plt.xlabel('$\log_{10}(power)$')
                plt.ylabel('proportion found at given frequency')
                plt.title(data_name + ' - power distributions')
                for f in findex:
                    plt.plot(h[1][1:] / np.log(10.0), hpwr[f, :], label='%7.5f Hz' % (freqs[f]))
                plt.legend(loc=3, fontsize=10)
                plt.savefig(savefig + '.power_spectra_distributions.png')

                # plot out the time series
                plt.figure(4)
                full_ts.peek()
                plt.savefig(savefig + '.full_ts_timeseries.png')
                plt.close('all')


do_lstsqr(dataroot='~/Data/AIA/',
          ldirroot='~/ts/pickle/',
          sfigroot='~/ts/img/',
          scsvroot='~/ts/csv/',
          corename='shutdownfun3_6hr',
          sunlocation='disk',
          fits_level='1.5',
          waves=['193', '211', '131'],
          regions=['moss', 'sunspot', 'qs', 'loopfootpoints'],
          windows=['hanning'],
          manip='relative')
"""

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
                 title=data_name + ' - '+ 'Bayesian/MCMC fit')
plt.savefig(savefig + '.mcmc_fit_with_stochastic_estimate.png')
"""