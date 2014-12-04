"""
USE THIS ONE!
CURRENT version
Specifies a directory of AIA files and uses a least-squares fit to generate
some fit estimates.
"""
import os
from matplotlib import rc_file
matplotlib_file = '~/ts/rednoise/py/matplotlibrc_paper1.rc'
rc_file(os.path.expanduser(matplotlib_file))
import matplotlib.pyplot as plt

import numpy as np
import tsutils
from timeseries import TimeSeries
# Curve fitting routine
from scipy.optimize import curve_fit

# Tests for normality
from scipy.stats import shapiro, anderson
from scipy.stats import spearmanr as Spearmanr
from statsmodels.graphics.gofplots import qqplot
#from rnsimulation import SimplePowerLawSpectrumWithConstantBackground, TimeSeriesFromPowerSpectrum
#from rnfit2 import Do_MCMC, rnsave
#from pymcmodels import single_power_law_with_constant_not_normalized
import cPickle as pickle
import aia_specific
import aia_plaw
from paper1 import log_10_product, tsDetails, s3min, s5min, s_U68, s_U95, s_L68, s_L95, figure_label_2by4
from paper1 import prettyprint, csv_timeseries_write, pkl_write, power_distribution_details
from paper1 import descriptive_stats
import scipy
from scipy.interpolate import interp1d
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


def independence_measures(a, b):
    # Calculate the cross-correlation coefficient
    ccvalue = np.correlate(cornorm(a, np.size(a)), cornorm(b, 1.0), mode='full')

    # Calculate normalized covariance matrix
    an = cornorm(a, 1.0)
    bn = cornorm(b, 1.0)
    covariance = np.cov(an, bn) / np.sqrt(np.var(an) * np.var(bn))

    # Calculate the spearman rho
    spearman = Spearmanr(a, b)

    return ccvalue, covariance, spearman


def get_cross_spectrum(a, b, wsize=10):
    a_fft = np.fft.fft(a)
    b_fft = np.fft.fft(b)
    ab_fft = np.conjugate(a_fft) * b_fft
    return tsutils.movingaverage(ab_fft, wsize)


def get_coherence(a, b, wsize=10):
    """ Calculate the coherence of two time series """
    a_pwr = np.abs(get_cross_spectrum(a, a, wsize=wsize))
    #
    b_pwr = np.abs(get_cross_spectrum(b, b, wsize=wsize))
    #
    ab_pwr = np.abs(get_cross_spectrum(a, b, wsize=wsize))
    #
    return (ab_pwr ** 2) / (a_pwr * b_pwr)


def calculate_histograms(nposfreq, pwr, bins, mask=None):
    # number of histogram bins
    hpwr = np.zeros((nposfreq, bins))
    for f in range(0, nposfreq):
        if mask == None:
            h = np.histogram(pwr[:, :, f], bins=bins, range=[np.min(pwr), np.max(pwr)])
            hpwr[f, :] = h[0] / (1.0 * np.sum(h[0]))
        else:
            results = []
            for i in range(0, mask.shape[1]):
                for j in range(0, mask.shape[0]):
                    results.append(pwr[j, i, f])
            results = np.asarray(results)
            h = np.histogram(results, bins=bins, range=[np.min(results), np.max(results)])
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
    return h[1], hpwr, lim


# Apply the manipulation function
def ts_manip(d, manip):
    dcopy = d.copy()
    if manip == 'relative':
        dmean = np.mean(dcopy)
        dcopy = dcopy / dmean - 1
    return dcopy, dmean


# Apply the window
def ts_apply_window(d, win):
    return d * win


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


def evenly_sampled(times, expected=12.0, absolute=0.1):
    return np.all(np.abs(times[1:] - times[0:-1] - expected) <= absolute)


# Main analysis loop
dataroot = '~/Data/AIA/'
ldirroot = '~/ts/pickle_cc_False_dr_False/'
sfigroot = '~/ts/img_cc_False_dr_False/'
scsvroot = '~/ts/csv_cc_False_dr_False/'
corename = 'study2'
sunlocation = 'equatorial'
sunlocation = 'spoca665'
fits_level = '1.5'
waves = ['171', '193']
regions = ['R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7']
waves = ['171']
regions = ['R4', 'R5', 'R6', 'R7']
windows = ['hanning']
manip = 'relative'
savefig_format = 'png'
freqfactor = [1000.0, 'mHz']
normalization_type = 'total_power'
sunday_name = {}
for region in regions:
    sunday_name[region] = region
neighbour = 'nearest'

five_min = freqfactor[0] * 1.0 / 300.0
three_min = freqfactor[0] * 1.0 / 180.0

coherence_wsize = 3

# main loop
for iwave, wave in enumerate(waves):
    # Now that the loading and saving locations are seot up, proceed with
    # the analysis.
    for iregion, region in enumerate(regions):
        # Create the branches in order
        branches = [corename, sunlocation, fits_level, wave, region]

        # Set up the roots we are interested in
        roots = {"pickle": ldirroot,
                 "image": sfigroot,
                 "csv": scsvroot}

        # Data and save locations are based here
        locations = aia_specific.save_location_calculator(roots,
                                     branches)

        # set the saving locations
        sfig = locations["image"]
        scsv = locations["csv"]

        # Identifier
        ident = aia_specific.ident_creator(branches)

        # Go through all the windows
        for iwindow, window in enumerate(windows):
            # General notification that we have a new data-set
            prettyprint('Loading New Data')
            # Which wavelength?
            print('Wave: ' + wave + ' (%i out of %i)' % (iwave + 1, len(waves)))
            # Which region
            print('Region: ' + region + ' (%i out of %i)' % (iregion + 1, len(regions)))
            # Which window
            print('Window: ' + window + ' (%i out of %i)' % (iwindow + 1, len(windows)))

            # Update the region identifier
            region_id = '.'.join((ident, window, manip))

            # Load the data
            pkl_location = locations['pickle']
            ifilename = ident + '.mapcube'
            pkl_file_location = os.path.join(pkl_location, ifilename + '.pickle')
            print('Loading ' + pkl_file_location)
            pkl_file = open(pkl_file_location, 'rb')
            dc = pickle.load(pkl_file).as_array()
            time_information = pickle.load(pkl_file)
            pkl_file.close()

            # Get some properties of the datacube
            ny = dc.shape[0]
            nx = dc.shape[1]
            nt = dc.shape[2]
            tsdetails = tsDetails(nx, ny, nt)

            # Should be evenly sampled data.  It not, then resample to get
            # evenly sampled data
            t = time_information["time_in_seconds"]
            dt = 12.0
            is_evenly_sampled = evenly_sampled(t, expected=dt, absolute=dt / 100.0)
            if not(is_evenly_sampled):
                print('Resampling to an even time cadence.')
                dt = (t[-1] - t[0]) / (1.0 * (nt - 1))
                evenly_sampled_t = np.arange(0, nt) * dt
                for iii in range(0, nx):
                    for jjj in range(0, ny):
                        f = interp1d(t, dc[iii, jjj, :])
                        dc[iii, jjj, :] = f(evenly_sampled_t)
                t = evenly_sampled_t

            # Define an array to store the analyzed data
            dc_analysed = np.zeros_like(dc)

            # calculate a window function
            win, dummy_winname = DefineWindow(window, nt)

            # Create the name for the data
            #data_name = wave + ' (' + fits_level + winname + ', ' + manip + '), ' + region
            #data_name = region_id
            if region in sunday_name:
                data_name = 'AIA ' + str(wave) + '$\AA$ : ' + sunday_name[region]
            else:
                data_name = 'AIA ' + str(wave) + '$\AA$ : ' + region

            # Create a location to save the figures
            savefig = os.path.join(os.path.expanduser(sfig), window, manip)
            if not(os.path.isdir(savefig)):
                os.makedirs(savefig)
            savefig = os.path.join(savefig, region_id)

            # Create a time series object
            t = dt * np.arange(0, nt)
            tsdummy = TimeSeries(t, t)
            freqs_original = tsdummy.PowerSpectrum.frequencies.positive
            freqs = freqfactor[0] * freqs_original
            posindex = np.fft.fftfreq(nt, dt) > 0.0 #tsdummy.PowerSpectrum.frequencies.posindex
            iobs = np.zeros(tsdummy.PowerSpectrum.ppower.shape)
            logiobs = np.zeros(tsdummy.PowerSpectrum.ppower.shape)
            logiobsDC = np.zeros(tsdummy.PowerSpectrum.ppower.shape)
            nposfreq = len(iobs)
            nfreq = tsdummy.PowerSpectrum.frequencies.nfreq

            # storage
            pwr = np.zeros((ny, nx, nposfreq))
            logpwr = np.zeros_like(pwr)
            doriginal = np.zeros_like(t)
            dmanip = np.zeros_like(t)
            fft_transform = np.zeros((ny, nx, nfreq), dtype=np.complex64)
            summed_intensity = 0.0
            logpwr_normalization = 0.0
            mask = np.zeros((ny, nx))

            npixel = 0
            for i in range(0, nx):
                for j in range(0, ny):

                    # Get the next time-series
                    d = dc[j, i, :].flatten()

                    # Fix the data for any non-finite entries
                    d = tsutils.fix_nonfinite(d)
                    dkeep = d.copy()

                    # Only consider non-zero data
                    if np.sum(d) > 0.0:
                        # Count the number of pixels that have non-zero data
                        npixel = npixel + 1
                        mask[j, i] = True

                        # Sum up all the original data
                        doriginal = doriginal + d

                        # Remove the mean
                        #if manip == 'submean':
                        #    d = d - np.mean(d)

                        # Basic rescaling of the time-series
                        d, dmean = ts_manip(d, manip)

                        # Sum up the intensity
                        summed_intensity = summed_intensity + dmean

                        # Sum up all the manipulated data
                        dmanip = dmanip + d

                        # Multiply the data by the apodization window
                        d = ts_apply_window(d, win)

                        # Keep the analyzed data cube
                        dc_analysed[j, i, :] = d

                        # Form a time series object.  Handy for calculating the Fourier power at all non-zero frequencies
                        ts = TimeSeries(t, d)

                        # Define the Fourier power we are analyzing
                        this_power = (np.abs(np.fft.fft(d)) ** 2)[posindex] / (1.0 * nt)#ts.PowerSpectrum.ppower

                        # Get the total Fourier power
                        iobs = iobs + this_power

                        # Store the individual Fourier power
                        pwr[j, i, :] = this_power

                        # Sum up the log of the Fourier power - equivalent to doing the geometric mean
                        logiobs = logiobs + np.log(this_power)

                        # Store the individual log Fourier power
                        logpwr[j, i, :] = np.log(this_power)

                        # Store the normalization factor
                        logpwr_normalization = logpwr_normalization + 2 * np.log(dmean)

                        # Get the FFT transform values and store them
                        fft_transform[j, i, :] = ts.fft_transform

            ###############################################################
            # Post-processing of the data products
            # Limits to the data
            dc_analysed_minmax = (np.min(dc_analysed), np.max(dc_analysed))

            # Original data: average
            doriginal = doriginal / (1.0 * npixel)

            # Manipulated data: average
            dmanip = dmanip / (1.0 * npixel)

            # Account for potentially different numbers of pixels
            logpwr_normalization = logpwr_normalization / (1.0 * npixel)

            # Show some information
            print('Number of non-zero pixels = %i' % (npixel))
            print('Size of containing region in pixels = %i' % (nx * ny))

            # Average of the analyzed time-series and create a time series
            # object
            full_data = np.mean(dc_analysed, axis=(0, 1))
            full_ts = TimeSeries(t, full_data)
            full_ts.name = data_name
            full_ts.label = 'average analyzed emission ' + tsdetails

            # Time series of the average original data
            doriginal, _ = ts_manip(doriginal, manip)
            #doriginal = ts_apply_window(d, win)
            doriginal_ts = TimeSeries(t, doriginal)
            doriginal_ts.name = data_name
            doriginal_ts.label = 'average summed emission ' + tsdetails

            # Fourier power: average over all the pixels
            iobs = iobs / (1.0 * npixel)

            # Fourier power: standard deviation over all the pixels
            sigma = np.std(pwr, axis=(0, 1))

            # Logarithmic power: average over all the pixels
            logiobs = logiobs / (1.0 * npixel)

            # Normalize relative to the first region looked at.
            if region == regions[0]:
                # Divide also by the summed intensity
                logpwr_normalization_keep = logpwr_normalization
                # normalize relative to the power in the lowest frequency
                if normalization_type == 'lowest_frequency':
                    normalization_region_power = logiobs[0]
                # normalize relative to the total power
                if normalization_type == 'total_power':
                    normalization_region_power = np.log(np.sum(np.exp(logiobs)))
                # normalize relative to the mean power
                if normalization_type == 'mean_power':
                    normalization_region_power = np.log(np.mean(np.exp(logiobs)))
            logiobs = logiobs + logpwr_normalization
            logpwr = logpwr + logpwr_normalization

            # Logarithmic power: standard deviation over all pixels
            logsigma = np.std(logpwr, axis=(0, 1))

            ###############################################################
            # Power spectrum analysis: arithmetic mean approach
            # Normalize the frequency.
            xnorm = tsdummy.PowerSpectrum.frequencies.positive[0]
            x = freqs_original / freqs_original[0]

            # Fourier power: fit a power law to the arithmetic mean of the
            # Fourier power

            #answer = curve_fit(aia_plaw_fit.PowerLawPlusConstant, x, iobs, sigma=sigma, p0=pguess)
            answer, error = aia_plaw.do_fit(x, iobs, aia_plaw.PowerLawPlusConstant, sigma=sigma)

            # Get the fit parameters out and calculate the best fit
            param = answer[0, 0, :]
            bf = aia_plaw.PowerLawPlusConstant(x,
                                               answer[0, 0, 0],
                                               answer[0, 0, 1],
                                               answer[0, 0, 2])

            # Error estimate for the power law index
            nerr = np.sqrt(error[0, 0, 1])

            ###############################################################
            # Estimate the correlation distance in the region.  This is
            # done by calculating the cross-correlation coefficient between
            # two randomly selected pixels in the region.  If we do this
            # often enough then we can estimate the distance at which the
            # the cross-correlation between two pixels is zero.  This
            # length-scale is then squared to get the estimated area that
            # contains a statistically independent time-series.  If we
            # divide the number of pixels in the region by this estimated
            # area then we get an estimate of the number of independent
            # time series in the region.
            prettyprint('Calculate pixel by pixel independence measures')

            def cornorm(a, norm):
                return (a - np.mean(a)) / (np.std(a) * norm)

            def exponential_decay(x, A, tau):
                return A * np.exp(-x / tau)

            def exponential_decay2(x, A1, tau1, A2, tau2):
                return A1 * np.exp(-x / tau1) + A2 * np.exp(-x / tau2)

            def exponential_decay3(x, A1, tau1, const):
                return A1 * np.exp(-x / tau1) + const

            def linear(x, c, m):
                return -m * x + c

            nsample = np.min(np.asarray([8 * npixel, 10000]))
            npicked = 0
            lag = 1
            distance = []
            cclag = []
            cc0 = []
            ccmax = []
            covar = []
            spearman = []
            logpwr_cclag = []
            logpwr_cc0 = []
            logpwr_ccmax = []
            logpwr_covar = []
            logpwr_spearman = []
            coher_array = np.zeros((nsample, nposfreq))
            while npicked < nsample:
                if neighbour == 'nearest':
                # Pick a location
                    loc1 = (np.random.randint(1, ny - 1), np.random.randint(1, nx - 1))
                    # Find a nearest neighbour
                    rchoice = np.random.randint(0, 8)
                    if rchoice == 0:
                        xdifference = -1
                        ydifference = -1
                    elif rchoice == 1:
                        xdifference = -1
                        ydifference = 0
                    elif rchoice == 2:
                        xdifference = -1
                        ydifference = 1
                    elif rchoice == 3:
                        xdifference = 0
                        ydifference = 1
                    elif rchoice == 4:
                        xdifference = 1
                        ydifference = 1
                    elif rchoice == 5:
                        xdifference = 1
                        ydifference = 0
                    elif rchoice == 6:
                        xdifference = 1
                        ydifference = -1
                    elif rchoice == 7:
                        xdifference = 0
                        ydifference = -1
                    loc2 = (loc1[0] + ydifference, loc1[1] + xdifference)

                if neighbour == 'random':
                    # Pick a random location
                    loc1 = (np.random.randint(0, ny), np.random.randint(0, nx))
                    loc2 = (np.random.randint(0, ny), np.random.randint(0, nx))
                    while (loc2[0] == loc1[0]) and (loc2[1] == loc1[1]):
                        loc2 = (np.random.randint(0, ny), np.random.randint(0, nx))

                # Get the time series
                ts1 = dc_analysed[loc1[0], loc1[1], :]
                ts2 = dc_analysed[loc2[0], loc2[1], :]

                if (np.sum(ts1) > 0) and (np.sum(ts2) > 0):

                    distance.append(np.sqrt(1.0 * (loc1[0] - loc2[0]) ** 2 + 1.0 * (loc1[1] - loc2[1]) ** 2))

                    # Calculate the independence measures
                    ccvalue, covariance, spearmanr = independence_measures(ts1, ts2)
                    cc0.append(ccvalue[nt - 1])
                    cclag.append(ccvalue[nt - 1 + lag])
                    ccmax.append(np.max(ccvalue))
                    covar.append(covariance[1, 0])
                    spearman.append(spearmanr[0])

                    # Calculate the coherence for each selected pair
                    coherence = get_coherence(ts1, ts2, wsize=coherence_wsize)
                    #if npicked == 0:
                    #    coher = coherence[posindex]
                    #else:
                    #    coher = coher + coherence[posindex]
                    coher_array[npicked, :] = coherence[posindex]

                    # Do the same thing for the log of the power spectra.
                    a = logpwr[loc1[0], loc1[1], :]
                    b = logpwr[loc2[0], loc2[1], :]
                    logpwr_ccvalue, logpwr_covariance, logpwr_spearmanr = independence_measures(a, b)
                    logpwr_cc0.append(logpwr_ccvalue[nposfreq - 1])
                    logpwr_cclag.append(logpwr_ccvalue[nposfreq - 1 + lag])
                    logpwr_ccmax.append(np.max(logpwr_ccvalue))
                    logpwr_spearman.append(logpwr_spearmanr[0])

                    # Advance the counter
                    npicked = npicked + 1

            # Average coherence
            coher = np.mean(coher_array, axis=0)

            # Standard deviation of the coherence
            coher_std = np.std(coher_array, axis=0)

            # Maximum coherence
            coher_max = np.max(coher_array, axis=0)

            # Histogram of coherence
            nbins = 100
            coher_hist = np.zeros((nposfreq, nbins))
            coher_95_hi = np.zeros(nposfreq)
            coher_mode = np.zeros_like(coher_95_hi)
            coher_mean = np.zeros_like(coher_95_hi)
            for jjj in range(0, nposfreq - 1):
                z = coher_array[:, jjj].flatten()
                h, _ = np.histogram(z, bins=nbins, range=(0.0, 1.0))
                coher_hist[jjj, :] = h / (1.0 * np.max(h))
                coher_ds = descriptive_stats(z)
                coher_95_hi[jjj] = coher_ds.cred[0.95][1]
                coher_mode[jjj] = coher_ds.mode
                coher_mean[jjj] = coher_ds.mean

            # All the pixels are all sqrt(2) pixels away from the
            # central pixel.  We treat them all as nearest neighbor.
            # What is the average correlation coefficient at the specified
            # lag?
            # Lag 0 cross correlation coefficient
            ccc_bins = 100
            cc0 = np.abs(np.asarray(cc0))
            cc0_ds = descriptive_stats(cc0, bins=ccc_bins)

            # Lag 'lag' cross correlation coefficient
            cclag = np.abs(np.asarray(cclag))
            cclag_ds = descriptive_stats(cclag, bins=ccc_bins)

            # Maximum cross correlation coefficient
            ccmax = np.abs(np.asarray(ccmax))
            ccmax_ds = descriptive_stats(ccmax, bins=ccc_bins)
            print '%s: Lag 0 cross correlation coefficient: mean, mode, median = %f, %f, %f' % (neighbour, cc0_ds.mean, cc0_ds.mode, cc0_ds.median)
            print '%s: Lag %i cross correlation coefficient: mean, mode, median = %f, %f, %f' % (neighbour, lag, cclag_ds.mean, cclag_ds.mode, cclag_ds.median)
            print '%s: Maximum cross correlation coefficient: mean, mode, median = %f, %f, %f' % (neighbour, ccmax_ds.mean, ccmax_ds.mode, ccmax_ds.median)

            # Maximum cross correlation coefficient of the log power
            logpwr_ccmax = np.abs(np.asarray(logpwr_ccmax))
            logpwr_ccmax_ds = descriptive_stats(logpwr_ccmax, bins=ccc_bins)
            print '%s: Maximum cross correlation coefficient, logpwr : mean, mode, median = %f, %f, %f' % (neighbour, logpwr_ccmax_ds.mean, logpwr_ccmax_ds.mode, logpwr_ccmax_ds.median)

            # Spearman rho details
            spearman = np.abs(np.asarray(spearman))
            spearman_ds = descriptive_stats(spearman, bins=ccc_bins)
            logpwr_spearman = np.abs(np.asarray(logpwr_spearman))
            logpwr_spearman_ds = descriptive_stats(logpwr_spearman, bins=ccc_bins)
            print '%s: Spearman rho: mean, mode, median = %f, %f, %f' % (neighbour, spearman_ds.mean, spearman_ds.mode, spearman_ds.median)
            print '%s: Spearman rho, logpwr: mean, mode, median = %f, %f, %f' % (neighbour, logpwr_spearman_ds.mean, logpwr_spearman_ds.mode, logpwr_spearman_ds.median)

            # Plot histograms of the three cross correlation coefficients
            ccc_bins = 100
            plt.figure(1)
            plt.hist(cc0, bins=ccc_bins, label='zero lag CCC', alpha=0.33)
            plt.axvline(cc0_ds.mean, label='cc0 mean %f' % cc0_ds.mean)
            plt.axvline(cc0_ds.mode, label='cc0 mode %f' % cc0_ds.mode)
            plt.axvline(cc0_ds.median, label='cc0 median %f' % cc0_ds.median)

            plt.hist(cclag, bins=ccc_bins, label='lag %i CCC' % (lag), alpha=0.33)
            plt.axvline(cclag_ds.mean, label='cclag mean %f' % cclag_ds.mean, linestyle=':')
            plt.axvline(cclag_ds.mode, label='cclag mode %f' % cclag_ds.mode, linestyle=':')
            plt.axvline(cclag_ds.median, label='cclag median %f' % cclag_ds.median, linestyle='--')

            plt.hist(ccmax, bins=ccc_bins, label='maximum CCC', alpha=0.33)
            plt.axvline(ccmax_ds.mean, label='ccmax mean %f' % ccmax_ds.mean, linestyle=':')
            plt.axvline(ccmax_ds.mode, label='ccmax mode %f' % ccmax_ds.mode, linestyle=':')
            plt.axvline(ccmax_ds.median, label='ccmax median %f' % ccmax_ds.median, linestyle='--')

            plt.hist(spearmanr, bins=ccc_bins, label='Spearman', alpha=0.33)
            plt.axvline(spearman_ds.mean, label='Spearman mean %f' % spearman_ds.mean, linestyle=':')
            plt.axvline(spearman_ds.mode, label='Spearman mode %f' % spearman_ds.mode, linestyle=':')
            plt.axvline(spearman_ds.median, label='Spearman median %f' % spearman_ds.median, linestyle='--')

            plt.xlabel('cross correlation coefficient (%s)' % (neighbour))
            plt.ylabel('number [%i samples]' % (npicked))
            plt.title(data_name + ' : Measures of cross correlation')
            plt.legend(fontsize=10, framealpha=0.5)
            plt.savefig(savefig + '.independence.cross_correlation_coefficients.%s.%s' % (neighbour, savefig_format))
            plt.close('all')

            # Plot histograms of the three independence coefficients
            plt.figure(1)
            plt.hist(1.0 - cc0, bins=ccc_bins, label='1 - |zero lag CCC|', alpha = 0.33)
            plt.axvline(1.0 - cc0_ds.mean,  linestyle=':', label='mean(1 - |zero lag CCC|) = %f' % (1.0 - cc0_ds.mean))

            plt.hist(1.0 - cclag, bins=ccc_bins, label='1 - |lag %i CCC|' % (lag), alpha = 0.33)
            plt.axvline(1.0 - cclag_ds.mean, linestyle='--', label='mean(1 - |lag %i CCC|) = %f' % (lag, 1.0 - cclag_ds.mean))

            plt.hist(1.0 - ccmax, bins=ccc_bins, label='1 - |max(CCC)|', alpha = 0.33)
            plt.axvline(1.0 - ccmax_ds.mean, label='mean(1 - |max(CCC)| = %f' % (1.0 - ccmax_ds.mean))

            plt.xlabel('independence coefficient (%s)' % (neighbour))
            plt.ylabel('number [%i samples]' % (npicked))
            plt.title(data_name + ' : measures of independence coefficient')
            plt.legend(fontsize=10, framealpha=0.5)
            plt.savefig(savefig + '.independence.independence_coefficients.%s.%s' % (neighbour, savefig_format))
            plt.close('all')

            # Plot histograms of the normalized covariance
            ccc_bins = 100
            plt.figure(1)
            plt.hist(np.asarray(covar), bins=ccc_bins, label='off diagonal covariance')
            plt.xlabel('normalized covariance')
            plt.ylabel('number [%i samples]' % (npicked))
            plt.title(data_name + ' : off diagonal covariance')
            plt.legend(fontsize=10, framealpha=0.5)
            plt.savefig(savefig + '.independence.covariance.%s' % (savefig_format))
            plt.close('all')

            # Plot the coherence measures
            plt.figure(1)
            plt.semilogx(freqs, coher, label='average coherence')
            plt.semilogx(freqs, coher_max, label='maximum coherence')
            plt.semilogx(freqs, coher_95_hi, label='high 95 % CI')
            plt.semilogx(freqs, coher_mode, label='mode')
            plt.xlabel('frequency (mHz)')
            plt.ylabel(data_name + ' : coherence')
            plt.title('Distribution of')
            plt.legend(fontsize=10, framealpha=0.5)
            plt.savefig(savefig + '.independence.coherence.%i.%s.%s' % (coherence_wsize, neighbour, savefig_format))
            plt.close('all')

            plt.figure(2)
            plt.imshow(coher_hist, origin='lower', aspect='auto', extent=(freqs[0], freqs[-1], 0, 1))
            plt.plot(freqs, coher_95_hi, label='high 95 % CI')
            plt.plot(freqs, coher_mode, label='mode')
            plt.plot(freqs, coher_max, label='maximum coherence')
            plt.plot(freqs, coher_mean, label='mean')
            plt.xlabel('frequency (mHz)')
            plt.ylabel('coherence')
            plt.title(data_name + ' : Coherence distribution (%s)' % (neighbour))
            plt.legend(fontsize=10, framealpha=0.5)
            plt.savefig(savefig + '.independence.coherence_histogram.%i.%s.%s' % (coherence_wsize, neighbour, savefig_format))
            plt.close('all')

            # Fourier power: get a Time series from the arithmetic sum of
            # all the time-series at every pixel, then apply the
            # manipulation and the window. Find the Fourier power
            # and do the fit.
            doriginal_ts_iobs = doriginal_ts.PowerSpectrum.ppower
            #answer_doriginal_ts = curve_fit(aia_plaw.LogPowerLawPlusConstant, x, np.log(doriginal_ts_iobs), p0=answer[0])
            #param_dts = answer_doriginal_ts[0]
            #bf_dts = np.exp(aia_plaw.LogPowerLawPlusConstant(x, param_dts[0], param_dts[1], param_dts[2]))
            #nerr_dts = np.sqrt(answer_doriginal_ts[1][1, 1])

            # -------------------------------------------------------------
            # Plots of power spectra: arithmetic means of summed emission
            # and summed power spectra
            ax = plt.subplot(111)

            # Set the scale type on each axis
            ax.set_xscale('log')
            ax.set_yscale('log')

            # Set the formatting of the tick labels
            xformatter = plt.FuncFormatter(log_10_product)
            ax.xaxis.set_major_formatter(xformatter)

            # Arithmetic mean of all the time series, then analysis
            ax.plot(freqs, doriginal_ts_iobs, color='r', label='sum over region')
            #ax.plot(freqs, bf_dts, color='r', linestyle="--", label='fit to sum over region n=%4.2f +/- %4.2f' % (param_dts[1], nerr_dts))

            # Arithmetic mean of the power spectra from each pixel
            ax.plot(freqs, iobs, color='b', label='arithmetic mean of power spectra from each pixel (Erlang distributed)')
            ax.plot(freqs, bf, color='b', linestyle="--", label='fit to arithmetic mean of power spectra from each pixel n=%4.2f +/- %4.2f' % (param[1], nerr))

            # Extra information for the plot
            ax.axvline(five_min, color=s5min.color, linestyle=s5min.linestyle, label=s5min.label)
            ax.axvline(three_min, color=s3min.color, linestyle=s3min.linestyle, label=s3min.label)
            #plt.axhline(1.0, color='k', label='average power')
            plt.xlabel('frequency (%s)' % (freqfactor[1]))
            plt.ylabel('normalized power [%i time series, %i samples each]' % (npixel, nt))
            plt.title(data_name + ' - arithmetic mean')
            #plt.grid()
            plt.legend(loc=3, fontsize=10, framealpha=0.5)
            #plt.text(freqs[0], 1.0, 'note: least-squares fit used, but data is not Gaussian distributed', fontsize=8)
            plt.savefig(savefig + '.arithmetic_mean_power_spectra.%s' % (savefig_format))
            plt.close('all')
            # -------------------------------------------------------------

            ###############################################################
            # Power spectrum analysis: geometric mean approach
            # ------------------------------------------------------------------------
            # Do the same thing over again, this time working with the log of the
            # normalized power.  This is the geometric mean

            # Fit the function to the log of the mean power
            #answer2 = curve_fit(aia_plaw.LogPowerLawPlusConstant, x, logiobs, sigma=logsigma, p0=answer[0])

            # Get the fit parameters out and calculate the best fit
            #param2 = answer2[0]
            #bf2 = np.exp(aia_plaw.LogPowerLawPlusConstant(x, param2[0], param2[1], param2[2]))

            # Error estimate for the power law index
            #nerr2 = np.sqrt(answer2[1][1, 1])

            # Create the histogram of all the log powers.  Histograms look normal-ish if
            # you take the logarithm of the power.  This suggests a log-normal distribution
            # of power in all frequencies

            # number of histogram bins
            # Calculate the probability density in each frequency bin.
            bins = 100
            bin_edges, hpwr, lim = calculate_histograms(nposfreq, logpwr, bins, mask=mask)
            histogram_loc = np.zeros(shape=(bins))
            for kk in range(0, bins):
                histogram_loc[kk] = 0.5 * (bin_edges[kk] + bin_edges[kk + 1])

            # -------------------------------------------------------------
            # plot some histograms of the log power at a small number of
            # frequencies.

            findex = []
            f_of_interest = [0.5 * five_min, five_min, three_min, 2 * three_min, 3 * three_min]
            hcolor = ['r', 'b', 'g', 'k', 'm']
            for thisf in f_of_interest:
                findex.append(np.unravel_index(np.argmin(np.abs(thisf - freqs)), freqs.shape)[0])
            plt.figure(3)
            plt.xlabel('$\log_{10}(power)$')
            plt.ylabel('proportion found')
            #plt.title(figure_label_2by4[wave][region] + data_name)
            plt.title(data_name)
            for jj, f in enumerate(findex):
                xx = histogram_loc / np.log(10.0)
                yy = hpwr[f, :]
                gfit = curve_fit(aia_plaw.GaussianShape2, xx, yy)
                #print gfit[0]
                plt.plot(xx, yy, color=hcolor[jj], label='%7.2f%s, $\sigma=$%3.2f' % (freqs[f], freqfactor[1], np.abs(gfit[0][2])))
                #plt.plot(xx, aia_plaw.GaussianShape2(xx, gfit[0][0], gfit[0][1],gfit[0][2]), color=hcolor[jj], linestyle='--')
            plt.legend(fontsize=12, framealpha=0.5, handletextpad=0.0, loc=2)
            plt.ylim(power_distribution_details()['ylim'][0], power_distribution_details()['ylim'][1])
            plt.xlim(power_distribution_details()['xlim'][0], power_distribution_details()['xlim'][1])
            plt.savefig(savefig + '.power_spectra_distributions.eps')
            plt.close('all')

            ###############################################################
            # Plot QQ-plots for the frequencies of interest

            plt.figure(3)
            for jj, f in enumerate(findex):
                qqplot(logpwr[:, :, findex[jj]].flatten(), line='s')
                plt.savefig(savefig + '.qqplot.%f.%s' % (f_of_interest[jj], savefig_format))
                plt.close('all')

            # Fit all the histogram curves to find the Gaussian width.
            # Stick with natural units to get the fit values which are
            # passed along to other programs.  Also, apply the Shapiro-Wilks
            # and Anderson-Darling tests for normality to test the assertion
            # that these distributions are approximately normal

            logiobs_width_fitted = np.zeros((nposfreq))
            error_logiobs_width_fitted = np.zeros_like(logiobs_width_fitted)
            logiobs_peak = np.zeros_like(logiobs_width_fitted)
            logiobs_peak_fitted = np.zeros_like(logiobs_width_fitted)
            logiobs_std = np.zeros_like(logiobs_width_fitted)
            logiobs_skew = np.zeros_like(logiobs_width_fitted)
            logiobs_kurtosis = np.zeros_like(logiobs_width_fitted)
            shapiro_wilks = []
            anderson_darling = []
            for jj, f in enumerate(freqs):
                all_logiobs_at_f = []
                for iy in range(0, ny):
                    for ix in range(0, nx):
                        if mask[iy, ix]:
                            all_logiobs_at_f.append(logpwr[iy, ix, jj])
                all_logiobs_at_f = np.asarray(all_logiobs_at_f)
                logiobs_std[jj] = np.std(all_logiobs_at_f)
                logiobs_skew[jj] = scipy.stats.skew(all_logiobs_at_f)
                logiobs_kurtosis[jj] = scipy.stats.kurtosis(all_logiobs_at_f)
                xx = histogram_loc
                yy = hpwr[jj, :]
                logiobs_peak[jj] = xx[np.argmax(yy)]
                # Apply the Shapiro-Wilks test and store the results
                nmzd = all_logiobs_at_f
                shapiro_wilks.append(shapiro(nmzd))
                # Apply the Anderson-Darling test and store the results
                anderson_darling_dist = 'norm'
                anderson_darling.append(anderson(nmzd, dist=anderson_darling_dist))
                # Try fitting a Gaussian shape. But maybe there is a better
                # shape that PyMC can sample from.  Other approach - extend
                # the wifth until 99% of the observed distribution lies
                # under the Gaussian distribution, keeping the amplitude of
                # the Gaussian distribution constant. This over-estimates
                # the error in the mean, and so is a bit more conservative.
                try:
                    p0 = [0, 0, 0]
                    p0[0] = np.max(yy)
                    p0[1] = xx[np.argmax(yy)]
                    p0[2] = 0.5#np.sqrt(np.mean(((p0[1] - xx) * yy) ** 2))
                    gfit = curve_fit(aia_plaw.GaussianShape2, xx, yy, p0=p0)
                    logiobs_width_fitted[jj] = np.abs(gfit[0][2])
                    error_logiobs_width_fitted[jj] = np.sqrt(np.abs(gfit[1][2, 2]))
                    logiobs_peak_fitted[jj] = gfit[0][1]
                except:
                    logiobs_width_fitted[jj] = None
                    error_logiobs_width_fitted[jj] = None
                    logiobs_peak_fitted[jj] = None

            # -------------------------------------------------------------
            # Plots of power spectra: geometric mean of power spectra at
            # each pixel
            ax = plt.subplot(111)

            # Set the scale type on each axis
            ax.set_xscale('log')

            # Set the formatting of the tick labels
            xformatter = plt.FuncFormatter(log_10_product)
            ax.xaxis.set_major_formatter(xformatter)

            # Geometric mean
            ax.plot(freqs, logiobs / np.log(10.0),  color='k', label='geometric mean')
            #ax.plot(freqs, bf2, color='k', label='best fit n=%4.2f +/- %4.2f' % (param2[1], nerr2))

            # Power at each frequency - distributions
            ax.plot(freqs, np.log10(lim[0, 0, :]), label=s_L68.label, color=s_L68.color, linewidth=s_L68.linewidth, linestyle=s_L68.linestyle)
            ax.plot(freqs, np.log10(lim[1, 0, :]), label=s_L95.label, color=s_L95.color, linewidth=s_L95.linewidth, linestyle=s_L95.linestyle)
            ax.plot(freqs, np.log10(lim[0, 1, :]), label=s_U68.label, color=s_U68.color, linewidth=s_U68.linewidth, linestyle=s_U68.linestyle)
            ax.plot(freqs, np.log10(lim[1, 1, :]), label=s_U95.label, color=s_U95.color, linewidth=s_U95.linewidth, linestyle=s_U95.linestyle)

            # Position of the fitted peak in each distribution
            #ax.plot(freqs, logiobs_peak_fitted / np.log(10.0),  color='m', label='fitted frequency')

            # Extra information for the plot
            ax.axvline(five_min, color=s5min.color, linestyle=s5min.linestyle, label=s5min.label)
            ax.axvline(three_min, color=s3min.color, linestyle=s3min.linestyle, label=s3min.label)
            plt.xlabel('frequency (%s)' % (freqfactor[1]))
            plt.ylabel('power [%i time series, %i samples each]' % (npixel, nt))
            plt.title(data_name + ' : geometric mean')
            plt.legend(loc=3, fontsize=10, framealpha=0.5)
            plt.savefig(savefig + '.geometric_mean_power_spectra_for_talk.%s' % (savefig_format))
            plt.close('all')
            # -------------------------------------------------------------

            # plot out the time series
            plt.figure(4)
            full_ts.peek()
            plt.savefig(savefig + '.full_ts_timeseries.%s' % (savefig_format))
            plt.close('all')

            # -------------------------------------------------------------
            # plot some histograms of the power at a small number of
            # frequencies.

            histogram_loc2, hpwr2, lim2 = calculate_histograms(nposfreq, pwr, 100)

            findex = []
            f_of_interest = [0.5 * five_min, five_min, three_min, 2 * three_min, 3 * three_min]
            for thisf in f_of_interest:
                findex.append(np.unravel_index(np.argmin(np.abs(thisf - freqs)), freqs.shape)[0])
            plt.figure(3)
            plt.xlabel('power')
            plt.ylabel('proportion found at given frequency')
            plt.title(data_name + ' - power distributions')
            for f in findex:
                xx = histogram_loc2[1:] / np.log(10.0)
                yy = hpwr2[f, :]
                plt.loglog(xx, yy, label='%7.2f %s' % (freqs[f], freqfactor[1]))
            plt.legend(loc=3, fontsize=10, framealpha=0.5)
            plt.savefig(savefig + '.notlog_power_spectra_distributions.%s' % (savefig_format))

            # plot out the time series
            plt.figure(4)
            full_ts.peek()
            plt.savefig(savefig + '.full_ts_timeseries.%s' % (savefig_format))
            plt.close('all')

            ###############################################################
            # Time series plots
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
            plt.savefig(savefig + '.all_analyzed_ts.%s' % (savefig_format))

            # Plot a histogram of the studied data at each time
            bins = 50
            hist_dc_analysed = np.zeros((bins, nt))
            for this_time in range(0, nt):
                hist_dc_analysed[:, this_time], bin_edges = np.histogram(dc_analysed[:, :, this_time], bins=bins, range=dc_analysed_minmax)
            hist_dc_analysed = hist_dc_analysed / (1.0 * npixel)
            plt.figure(12)
            plt.xlabel('time (seconds)')
            plt.ylabel('analyzed emission ' + tsdetails)
            plt.imshow(hist_dc_analysed, aspect='auto', origin='lower',
                       extent=(t[0], t[-1], dc_analysed_minmax[0], dc_analysed_minmax[1]))
            plt.colorbar()
            plt.title(data_name)
            plt.savefig(savefig + '.all_analyzed_ts_histogram.%s' % (savefig_format))

            ###############################################################
            # Fourier power plots
            # Plot all the analyzed FFTs
            plt.figure(11)
            for i in range(0, nx):
                for j in range(0, ny):
                    if np.sum(dc_analysed[j, i, :]) > 0:
                        ts = TimeSeries(t, dc_analysed[j, i, :])
                        plt.loglog(freqs, ts.PowerSpectrum.ppower)
            plt.loglog()
            plt.axvline(five_min, color=s5min.color, linestyle=s5min.linestyle, label=s5min.label)
            plt.axvline(three_min, color=s3min.color, linestyle=s3min.linestyle, label=s3min.label)
            plt.xlabel('frequency (%s)' % (freqfactor[1]))
            plt.ylabel('FFT power ' + tsdetails)
            plt.title(data_name)
            plt.savefig(savefig + '.all_analyzed_fft.%s' % (savefig_format))

            # Plot a histogram of the studied FFTs at each time
            bins = 50
            minmax = [np.min(logpwr), np.max(logpwr)]
            hist_dc_analysed_logpwr = np.zeros((bins, nposfreq))
            for this_freq in range(0, nposfreq):
                hist_dc_analysed_logpwr[:, this_freq], bin_edges = np.histogram(logpwr[:, :, this_freq], bins=bins, range=minmax)
            hist_dc_analysed_logpwr = hist_dc_analysed_logpwr / (1.0 * npixel)
            plt.figure(13)
            plt.xlabel('frequency (%s)' % (freqfactor[1]))
            plt.ylabel('FFT power ' + tsdetails)
            plt.imshow(hist_dc_analysed_logpwr, aspect='auto', origin='lower',
                       extent=(freqs[0], freqs[-1], np.exp(minmax[0]), np.exp(minmax[1])))
            plt.semilogy()
            plt.colorbar()
            plt.title(data_name)
            plt.savefig(savefig + '.all_analyzed_fft_histogram.%s' % (savefig_format))

            ###############################################################
            # Plot of the widths as a function of the frequency.
            plt.figure(14)
            plt.xlabel('frequency (%s)' % (freqfactor[1]))
            plt.ylabel('decades of frequency')
            plt.semilogx(freqs, (logiobs_width_fitted + error_logiobs_width_fitted) / np.log(10.0), label='+ error', linestyle='--')
            plt.semilogx(freqs, (logiobs_width_fitted - error_logiobs_width_fitted) / np.log(10.0), label='- error', linestyle='--')
            plt.semilogx(freqs, logiobs_width_fitted / np.log(10.0), label='estimated width')
            plt.semilogx(freqs, logiobs_std / np.log(10.0), label='standard deviation')
            plt.semilogx(freqs, (logiobs - logiobs_peak_fitted) / np.log(10.0), label='mean - fitted peak')
            plt.title(data_name + ' - distribution widths')
            plt.legend(loc=3, framealpha=0.3, fontsize=10)
            plt.savefig(savefig + '.distribution_width.logiobs.%s' % (savefig_format))

            ###############################################################
            # Plot the results of the skewness of the power distributions

            plt.figure(15)
            plt.xlabel('frequency (%s)' % (freqfactor[1]))
            plt.ylabel('skew')
            plt.semilogx(freqs, logiobs_skew, label='sample skew')
            plt.title(data_name + ': Fourier power distributions, skewness')
            plt.axhline(0.0, label='>0, right tailed: <0, left tailed', linestyle='--')
            plt.ylim(np.min([np.min(logiobs_skew), -0.5]), np.max([np.max(logiobs_skew), 0.5]))
            plt.legend(loc=3, framealpha=0.3, fontsize=10)
            plt.savefig(savefig + '.power_spectra_distributions.skewness.%s' % (savefig_format))
            plt.close('all')

            ###############################################################
            # Plot the results of the kurtosis of the power distributions

            plt.figure(15)
            plt.xlabel('frequency (%s)' % (freqfactor[1]))
            plt.ylabel('kurtosis')
            plt.semilogx(freqs, logiobs_kurtosis, label='sample kurtosis')
            plt.axhline(3.0, label='>3, leptokurtic (slender): <3, platykurtic (broad)', linestyle='--')
            plt.ylim(np.min([np.min(logiobs_kurtosis), 2.5]), np.max([np.max(logiobs_kurtosis), 3.5]))
            plt.title(data_name + ': Fourier power distributions, kurtosis')
            plt.legend(loc=3, framealpha=0.3, fontsize=10)
            plt.savefig(savefig + '.power_spectra_distributions.kurtosis.%s' % (savefig_format))
            plt.close('all')

            ###############################################################
            # Plot the results of the Shapiro-Wilks test for the Fourier
            # power distributions.  Low p-values reject the null hypothesis
            # of normality.
            """
            pvalue = np.asarray([result[1] for result in shapiro_wilks])
            plt.figure(15)
            plt.xlabel('frequency (%s)' % (freqfactor[1]))
            plt.ylabel('p-value [low values reject normality]')
            plt.loglog(freqs, pvalue, label='pvalue', linestyle='--')
            plt.title(data_name + ': Fourier power distributions, Shapiro-Wilks normality test')
            plt.legend(loc=3, framealpha=0.3, fontsize=10)
            plt.savefig(savefig + '.distribution_width.shapiro_wilks.%s' % (savefig_format))
            """
            ###############################################################
            # Plot the results of the Anderson-Darling test for the Fourier
            # power distributions.  If the AD statistic value is above a
            # particular critical value corresponding to a given
            # significance level, then the null hypothesis (in this case
            # normality) can be rejected at that significance level.
            """
            statvalue = np.asarray([result[0] for result in anderson_darling])
            crit = anderson_darling[0][1]
            significance = anderson_darling[0][2]
            plt.figure(16)
            plt.xlabel('frequency (%s)' % (freqfactor[1]))
            plt.ylabel('AD statistic value [' + anderson_darling_dist + ']')
            for jjj, sig in enumerate(significance):
                plt.axhline(crit[jjj], color='k')
                plt.text(freqs[1], crit[jjj], 'reject %s at %s %%' % (anderson_darling_dist, str(np.rint(sig))))
            plt.loglog(freqs, statvalue, label='AD statistic', linestyle='--')
            plt.title(data_name + ': Fourier power distributions, Anderson-Darling test')
            plt.legend(loc=3, framealpha=0.3, fontsize=10)
            plt.savefig(savefig + '.distribution_width.anderson_darling.%s.%s' % (anderson_darling_dist, savefig_format))
            """
            ###############################################################
            # Save various data products
            # Fourier Power of the analyzed data
            ofilename = region_id
            pkl_write(pkl_location,
                      'OUT.' + ofilename + '.fourier_power.pickle',
                      (freqs_original, pwr))

            # Analyzed data
            pkl_write(pkl_location,
                      'OUT.' + ofilename + '.dc_analysed.pickle',
                      (t, dc_analysed))

            # Fourier transform
            pkl_write(pkl_location,
                      'OUT.' + ofilename + '.fft_transform.pickle',
                      (freqs_original, fft_transform))

            # logarithm of the arithmetic mean of power spectra
            pkl_write(pkl_location,
                      'OUT.' + ofilename + '.iobs.pickle',
                      (freqs_original, np.log(iobs)))

            # Geometric mean of power spectra
            # (frequencies, number of pixels, mean logiobs ,gaussian fitted
            # logiobs, peak logiobs)
            pkl_write(pkl_location,
                      'OUT.' + ofilename + '.logiobs.pickle',
                      (freqs_original, npixel,
                       logiobs,
                       logiobs_peak_fitted,
                       logiobs_peak))

            # Widths of the power distributions
            #(frequencies
            pkl_write(pkl_location,
                      'OUT.' + ofilename + '.distribution_widths.pickle',
                      (freqs_original,
                       logiobs_std,
                       logiobs_width_fitted))

            # Error in the Widths of the power distributions
            pkl_write(pkl_location,
                      'OUT.' + ofilename + '.distribution_widths_error.pickle',
                      (freqs_original, error_logiobs_width_fitted))

            # Coherence quantities
            pkl_write(pkl_location,
                      'OUT.' + ofilename + '.coherence.' + str(coherence_wsize) + '.'  + neighbour + '.pickle',
                      (freqs_original, coher, coher_max, coher_mode,
                       coher_95_hi, coher_mean))

            # Correlation / independence quantities
            pkl_write(pkl_location,
                      'OUT.' + ofilename + '.correlative.' + neighbour + '.pickle',
                      (cc0_ds, cclag_ds, ccmax_ds, spearman_ds))

            # Correlation / independence quantities
            np.save(os.path.join(pkl_location, 'OUT.' + ofilename + '.correlative2.' + neighbour + '.npy'), ccmax)

            # Correlation / independence quantities
            np.save(os.path.join(pkl_location, 'OUT.' + ofilename + '.spearman.' + neighbour + '.npy'), spearman)

            # Correlation / independence quantities
            np.save(os.path.join(pkl_location, 'OUT.' + ofilename + '.logpwr_spearman.' + neighbour + '.npy'), logpwr_spearman)

            # Save the full time series to a CSV file
            csv_timeseries_write(os.path.join(os.path.expanduser(scsv), window, manip),
                                 '.'.join((data_name, 'average_analyzed_ts.csv')),
                                 (t, full_data))

            # Original data
            csv_timeseries_write(os.path.join(os.path.expanduser(scsv)),
                                 '.'.join((ident, 'average_original_ts.csv')),
                                 (t, doriginal))

            ###############################################################

"""
do_lstsqr(dataroot='~/Data/AIA/',
          ldirroot='~/ts/pickle/',
          sfigroot='~/ts/img/',
          scsvroot='~/ts/csv/',
          corename='study2',
          sunlocation='equatorial',
          fits_level='1.5',
          waves=['171', '193'],
          regions=['R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7'],
          windows=['hanning'],
          manip='relative')
"""
