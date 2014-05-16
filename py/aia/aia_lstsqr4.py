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
# Curve fitting routine
from scipy.optimize import curve_fit

# Tests for normality
from scipy.stats import shapiro
from scipy.stats import anderson
#from rnsimulation import SimplePowerLawSpectrumWithConstantBackground, TimeSeriesFromPowerSpectrum
#from rnfit2 import Do_MCMC, rnsave
#from pymcmodels import single_power_law_with_constant_not_normalized
import csv
import cPickle as pickle
import aia_specific
import aia_plaw
from paper1 import log_10_product, tsDetails, s3min, s5min, s_U68, s_U95, s_L68, s_L95, prettyprint
"""
font = {'family': 'normal',
        'weight': 'bold',
        'size': 12}

matplotlib.rc('font', **font)
matplotlib.rc('lines', linewidth=1)
matplotlib.rc('figure', figsize=(12.5, 10))
"""
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


def calculate_histograms(nposfreq, pwr, bins):
    # number of histogram bins
    hpwr = np.zeros((nposfreq, bins))
    for f in range(0, nposfreq):
        h = np.histogram(pwr[:, :, f], bins=bins, range=[np.min(pwr), np.max(pwr)])
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
    return dcopy


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


# Write out a pickle file
def pkl_write(location, fname, a):
    pkl_file_location = os.path.join(location, fname)
    print('Writing ' + pkl_file_location)
    pkl_file = open(pkl_file_location, 'wb')
    for element in a:
        pickle.dump(element, pkl_file)
    pkl_file.close()


# Write a CSV time series
def csv_timeseries_write(location, fname, a):
    # Get the time and the data out separately
    t = a[0]
    d = a[1]
    # Make the directory if it is not already
    if not(os.path.isdir(location)):
        os.makedirs(location)
    # full file name
    savecsv = os.path.join(location, fname)
    # Open and write the file
    ofile = open(savecsv, "wb")
    writer = csv.writer(ofile, delimiter=',')
    for i, dd in enumerate(d):
        writer.writerow([t[i], dd])
    ofile.close()


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
              manip='none',
              savefig_format='png',
              freqfactor=[1000.0, 'mHz'],
              sunday_name={"qs": "quiet Sun", "loopfootpoints": "loop footpoints"}):

    five_min = freqfactor[0] * 1.0 / 300.0
    three_min = freqfactor[0] * 1.0 / 180.0

    # main loop
    for iwave, wave in enumerate(waves):
        # Which wavelength?
        print('Wave: ' + wave + ' (%i out of %i)' % (iwave + 1, len(waves)))

        # Now that the loading and saving locations are seot up, proceed with
        # the analysis.
        for iregion, region in enumerate(regions):
            # Which region
            print('Region: ' + region + ' (%i out of %i)' % (iregion + 1, len(regions)))

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
                # Which window
                print('Window: ' + window + ' (%i out of %i)' % (iwindow + 1, len(windows)))

                # Update the region identifier
                region_id = '.'.join((ident, window, manip))

                # Load the data
                pkl_location = locations['pickle']
                ifilename = ident + '.datacube'
                pkl_file_location = os.path.join(pkl_location, ifilename + '.pickle')
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
                win, dummy_winname = DefineWindow(window, nt)

                # Create the name for the data
                #data_name = wave + ' (' + fits_level + winname + ', ' + manip + '), ' + region
                #data_name = region_id
                if region in sunday_name:
                    data_name = 'AIA ' + str(wave) + ', ' + sunday_name[region]
                else:
                    data_name = 'AIA ' + str(wave) + ', ' + region

                # Create a location to save the figures
                savefig = os.path.join(os.path.expanduser(sfig), window, manip)
                if not(os.path.isdir(savefig)):
                    os.makedirs(savefig)
                savefig = os.path.join(savefig, region_id)

                # Create a time series object
                dt = 12.0
                t = dt * np.arange(0, nt)
                tsdummy = TimeSeries(t, t)
                freqs = freqfactor[0] * tsdummy.PowerSpectrum.frequencies.positive
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

                for i in range(0, nx):
                    for j in range(0, ny):

                        # Get the next time-series
                        d = dc[j, i, :].flatten()

                        # Fix the data for any non-finite entries
                        d = tsutils.fix_nonfinite(d)
                        dkeep = d.copy()

                        # Sum up all the original data
                        doriginal = doriginal + d

                        # Remove the mean
                        #if manip == 'submean':
                        #    d = d - np.mean(d)

                        # Basic rescaling of the time-series
                        d = ts_manip(d, manip)

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

                        # Get the FFT transform values and store them
                        fft_transform[j, i, :] = ts.fft_transform

                ###############################################################
                # Post-processing of the data products
                # Limits to the data
                dc_analysed_minmax = (np.min(dc_analysed), np.max(dc_analysed))

                # Original data: average
                doriginal = doriginal / (1.0 * nx * ny)

                # Manipulated data: average
                dmanip = dmanip / (1.0 * nx * ny)

                # Average of the analyzed time-series and create a time series
                # object
                full_data = np.mean(dc_analysed, axis=(0, 1))
                full_ts = TimeSeries(t, full_data)
                full_ts.name = data_name
                full_ts.label = 'average analyzed emission ' + tsdetails

                # Time series of the average original data
                doriginal = ts_manip(doriginal, manip)
                doriginal = ts_apply_window(d, win)
                doriginal_ts = TimeSeries(t, doriginal)
                doriginal_ts.name = data_name
                doriginal_ts.label = 'average summed emission ' + tsdetails

                # Fourier power: average over all the pixels
                iobs = iobs / (1.0 * nx * ny)

                # Fourier power: standard deviation over all the pixels
                sigma = np.std(pwr, axis=(0, 1))

                # Logarithmic power: average over all the pixels
                logiobs = logiobs / (1.0 * nx * ny)

                # Logarithmic power: standard deviation over all pixels
                logsigma = np.std(logpwr, axis=(0, 1))

                ###############################################################
                # Power spectrum analysis: arithmetic mean approach
                # Normalize the frequency.
                xnorm = tsdummy.PowerSpectrum.frequencies.positive[0]
                x = freqs / (xnorm * freqfactor[0])

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
                prettyprint('Calculate pixel by pixel correlation')
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

                nsample = np.min(np.asarray([8 * nx * ny, 10000]))
                npicked = 0
                lag = 1
                cclag = []
                cc0 = []
                ccmax = []
                covar = []
                coher_array = np.zeros((nsample, nposfreq))
                while npicked < nsample:
                    # Pick a location
                    loc1 = (np.random.randint(1, ny - 1), np.random.randint(1, nx - 1))

                    # Find the nearest neighbour
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

                    # Get the time series
                    ts1 = dc_analysed[loc1[0], loc1[1], :]
                    ts2 = dc_analysed[loc2[0], loc2[1], :]

                    # Calculate the cross-correlation coefficient
                    ccvalue = np.correlate(cornorm(ts1, np.size(ts1)), cornorm(ts2, 1.0), mode='full')
                    cc0.append(ccvalue[nt - 1])
                    cclag.append(ccvalue[nt - 1 + lag])
                    ccmax.append(np.max(ccvalue))

                    # Calculate the normalized covariance matrix
                    ts1n = cornorm(ts1, 1.0)
                    ts2n = cornorm(ts2, 1.0)
                    covariance = np.cov(ts1n, ts2n) / np.sqrt(np.var(ts1n) * np.var(ts2n))
                    covar.append(covariance[1, 0])

                    # Calculate the coherence for each selected pair, and add
                    # them up to find the average
                    coherence = get_coherence(ts1, ts2)
                    #if npicked == 0:
                    #    coher = coherence[posindex]
                    #else:
                    #    coher = coher + coherence[posindex]
                    coher_array[npicked, :] = coherence[posindex]

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
                for jjj in range(0, nposfreq - 1):
                    h, _ = np.histogram(coher_array[:, jjj], bins=nbins, range=(0.0, 1.0))
                    coher_hist[jjj, :] = h / (1.0 * np.max(h))

                # All the pixels are all sqrt(2) pixels away from the
                # central pixel.  We treat them all as nearest neighbor.
                # What is the average correlation coefficient at the specified
                # lag?
                ccc0 = np.mean(np.abs(np.asarray(cc0)))
                ccclag = np.mean(np.abs(np.asarray(cclag)))
                cccmax = np.mean(np.abs(np.asarray(ccmax)))
                print 'Average lag 0 cross correlation coefficient = %f' % (ccc0)
                print 'Average lag %i cross correlation coefficient = %f' % (lag, ccclag)
                print 'Average maximum cross correlation coefficient = %f' % (cccmax)

                # Plot histograms of the three cross correlation coefficients
                ccc_bins = 100
                plt.figure(1)
                plt.hist(np.asarray(cc0), bins=ccc_bins, label='zero lag CCC', alpha = 0.33)
                plt.hist(np.asarray(cclag), bins=ccc_bins, label='lag %i CCC' % (lag), alpha = 0.33)
                plt.hist(np.asarray(ccmax), bins=ccc_bins, label='maximum CCC', alpha = 0.33)
                plt.xlabel('cross correlation coefficient')
                plt.ylabel('number [%i samples]' % (npicked))
                plt.title(data_name + ' : Measures of cross correlation')
                plt.legend(fontsize=10, framealpha=0.5)
                plt.savefig(savefig + '.cross_correlation_coefficients.%s' % (savefig_format))
                plt.close('all')

                # Plot histograms of the three independence coefficients
                plt.figure(1)
                plt.hist(1.0 - np.abs(np.asarray(cc0)), bins=ccc_bins, label='1 - |zero lag CCC|', alpha = 0.33)
                cc0_mean = np.mean(1.0 - np.abs(np.asarray(cc0)))
                plt.axvline(cc0_mean, linestyle=':', label='mean(1 - |zero lag CCC|) = %f' % (cc0_mean))

                plt.hist(1.0 - np.abs(np.asarray(cclag)), bins=ccc_bins, label='1 - |lag %i CCC|' % (lag), alpha = 0.33)
                cclag_mean = np.mean(1.0 - np.abs(np.asarray(cclag)))
                plt.axvline(cclag_mean, linestyle='--', label='mean(1 - |lag %i CCC|) = %f' % (lag, cclag_mean))

                plt.hist(1.0 - np.abs(np.asarray(ccmax)), bins=ccc_bins, label='1 - |max(CCC)|', alpha = 0.33)
                ccmax_mean = np.mean(1.0 - np.abs(np.asarray(ccmax)))
                plt.axvline(ccmax_mean, label='mean(1 - |max(CCC)| = %f' % (ccmax_mean))

                plt.xlabel('independence coefficient')
                plt.ylabel('number [%i samples]' % (npicked))
                plt.title(data_name + ' : measures of independence coefficient')
                plt.legend(fontsize=10, framealpha=0.5)
                plt.savefig(savefig + '.independence_coefficients.%s' % (savefig_format))
                plt.close('all')

                # Plot histograms of the normalized covariance
                ccc_bins = 100
                plt.figure(1)
                plt.hist(np.asarray(covar), bins=ccc_bins, label='off diagonal covariance')
                plt.xlabel('normalized covariance')
                plt.ylabel('number [%i samples]' % (npicked))
                plt.title(data_name + ' : off diagonal covariance')
                plt.legend(fontsize=10, framealpha=0.5)
                plt.savefig(savefig + '.covariance.%s' % (savefig_format))
                plt.close('all')

                # Plot the coherence measures
                plt.figure(1)
                plt.semilogx(freqs, coher, label='average coherence')
                plt.semilogx(freqs, coher + coher_std, label='average coherence + std')
                plt.semilogx(freqs, coher - coher_std, label='average coherence - std')
                plt.semilogx(freqs, coher_max, label='maximum coherence')
                plt.xlabel('frequency (mHz)')
                plt.ylabel(data_name + ' : coherence')
                plt.title('Distribution of')
                plt.legend(fontsize=10, framealpha=0.5)
                plt.savefig(savefig + '.coherence.%s' % (savefig_format))
                plt.close('all')

                plt.figure(2)
                plt.imshow(coher_hist, origin='lower', aspect='auto', extent=(freqs[0], freqs[-1], 0, 1))
                plt.xlabel('frequency (mHz)')
                plt.ylabel('coherence')
                plt.title(data_name + ' : Coherence distribution')
                plt.savefig(savefig + '.coherence_histogram.%s' % (savefig_format))
                plt.close('all')

                # Fourier power: get a Time series from the arithmetic sum of
                # all the time-series at every pixel, then apply the
                # manipulation and the window. Find the Fourier power
                # and do the fit.
                doriginal_ts_iobs = doriginal_ts.PowerSpectrum.ppower
                answer_doriginal_ts = curve_fit(aia_plaw.LogPowerLawPlusConstant, x, np.log(doriginal_ts_iobs), p0=answer[0])
                param_dts = answer_doriginal_ts[0]
                bf_dts = np.exp(aia_plaw.LogPowerLawPlusConstant(x, param_dts[0], param_dts[1], param_dts[2]))
                nerr_dts = np.sqrt(answer_doriginal_ts[1][1, 1])

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
                ax.plot(freqs, bf_dts, color='r', linestyle="--", label='fit to sum over region n=%4.2f +/- %4.2f' % (param_dts[1], nerr_dts))

                # Arithmetic mean of the power spectra from each pixel
                ax.plot(freqs, iobs, color='b', label='arithmetic mean of power spectra from each pixel (Erlang distributed)')
                ax.plot(freqs, bf, color='b', linestyle="--", label='fit to arithmetic mean of power spectra from each pixel n=%4.2f +/- %4.2f' % (param[1], nerr))

                # Extra information for the plot
                ax.axvline(five_min, color=s5min.color, linestyle=s5min.linestyle, label=s5min.label)
                ax.axvline(three_min, color=s3min.color, linestyle=s3min.linestyle, label=s3min.label)
                #plt.axhline(1.0, color='k', label='average power')
                plt.xlabel('frequency (%s)' % (freqfactor[1]))
                plt.ylabel('normalized power [%i time series, %i samples each]' % (nx * ny, nt))
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
                answer2 = curve_fit(aia_plaw.LogPowerLawPlusConstant, x, logiobs, sigma=logsigma, p0=answer[0])

                # Get the fit parameters out and calculate the best fit
                param2 = answer2[0]
                bf2 = np.exp(aia_plaw.LogPowerLawPlusConstant(x, param2[0], param2[1], param2[2]))

                # Error estimate for the power law index
                nerr2 = np.sqrt(answer2[1][1, 1])

                # Create the histogram of all the log powers.  Histograms look normal-ish if
                # you take the logarithm of the power.  This suggests a log-normal distribution
                # of power in all frequencies

                # number of histogram bins
                # Calculate the probability density in each frequency bin.
                bins = 100
                bin_edges, hpwr, lim = calculate_histograms(nposfreq, logpwr, bins)
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
                plt.ylabel('proportion found at given frequency')
                plt.title(data_name + ' : power distributions')
                for jj, f in enumerate(findex):
                    xx = histogram_loc / np.log(10.0)
                    yy = hpwr[f, :]
                    gfit = curve_fit(aia_plaw.GaussianShape2, xx, yy)
                    #print gfit[0]
                    plt.plot(xx, yy, color=hcolor[jj], label='%7.2f %s, $\sigma=$ %3.2f' % (freqs[f], freqfactor[1], np.abs(gfit[0][2])))
                    plt.plot(xx, aia_plaw.GaussianShape2(xx, gfit[0][0], gfit[0][1],gfit[0][2]), color=hcolor[jj], linestyle='--')
                plt.legend(loc=3, fontsize=10, framealpha=0.5)
                plt.savefig(savefig + '.power_spectra_distributions.%s' % (savefig_format))
                plt.close('all')

                # Fit all the histogram curves to find the Gaussian width.
                # Stick with natural units to get the fit values which are
                # passed along to other programs.  Also, apply the Shapiro-Wilks
                # and Anderson-Darling tests for normality to test the assertion
                # that these distributions are approximately normal
                logiobs_distrib_width = np.zeros((nposfreq))
                error_logiobs_distrib_width = np.zeros_like(logiobs_distrib_width)
                iobs_peak = np.zeros_like(logiobs_distrib_width)
                logiobs_peak_location = np.zeros_like(logiobs_distrib_width)
                logiobs_std = np.zeros_like(logiobs_distrib_width)
                shapiro_wilks = []
                anderson_darling = []
                for jj, f in enumerate(freqs):
                    all_logiobs_at_f = logpwr[:, :, jj]
                    logiobs_std[jj] = np.std(all_logiobs_at_f)
                    xx = histogram_loc
                    yy = hpwr[jj, :]
                    iobs_peak[jj] = xx[np.argmax(yy)]
                    # Apply the Shapiro-Wilks test and store the results
                    shapiro_wilks.append(shapiro(all_logiobs_at_f))
                    # Apply the Anderson-Darling test and store the results
                    anderson_darling.append(anderson(all_logiobs_at_f, dist='norm'))
                    # Try the fit
                    try:
                        p0 = [0, 0, 0]
                        p0[0] = np.max(yy)
                        p0[1] = xx[np.argmax(yy)]
                        p0[2] = 0.5#np.sqrt(np.mean(((p0[1] - xx) * yy) ** 2))
                        gfit = curve_fit(aia_plaw.GaussianShape2, xx, yy, p0=p0)
                        logiobs_distrib_width[jj] = np.abs(gfit[0][2])
                        error_logiobs_distrib_width[jj] = np.sqrt(np.abs(gfit[1][2, 2]))
                        logiobs_peak_location[jj] = gfit[0][1]
                    except:
                        logiobs_distrib_width[jj] = None
                        error_logiobs_distrib_width[jj] = None
                        logiobs_peak_location[jj] = None

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
                #ax.plot(freqs, logiobs_peak_location / np.log(10.0),  color='m', label='fitted frequency')

                # Extra information for the plot
                ax.axvline(five_min, color=s5min.color, linestyle=s5min.linestyle, label=s5min.label)
                ax.axvline(three_min, color=s3min.color, linestyle=s3min.linestyle, label=s3min.label)
                plt.xlabel('frequency (%s)' % (freqfactor[1]))
                plt.ylabel('power [%i time series, %i samples each]' % (nx * ny, nt))
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
                """
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
                """
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
                hist_dc_analysed = hist_dc_analysed / (1.0 * nx * ny)
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
                hist_dc_analysed_logpwr = hist_dc_analysed_logpwr / (1.0 * nx * ny)
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
                plt.semilogx(freqs, (logiobs_distrib_width + error_logiobs_distrib_width) / np.log(10.0), label='+ error', linestyle='--')
                plt.semilogx(freqs, (logiobs_distrib_width - error_logiobs_distrib_width) / np.log(10.0), label='- error', linestyle='--')
                plt.semilogx(freqs, logiobs_distrib_width / np.log(10.0), label='estimated width')
                plt.semilogx(freqs, logiobs_std / np.log(10.0), label='standard deviation')
                plt.semilogx(freqs, (logiobs - logiobs_peak_location) / np.log(10.0), label='mean - fitted peak')
                plt.title(data_name + ' - distribution widths')
                plt.legend(loc=3, framealpha=0.3, fontsize=10)
                plt.savefig(savefig + '.logiobs_distribution_width.%s' % (savefig_format))

                ###############################################################
                # Plot the results of the Shapiro-Wilks test for the Fourier
                # power distributions.  Low p-values reject the null hypothesis
                # of normality.
                pvalue = np.asarray([result[1] for result in shapiro_wilks])
                plt.figure(15)
                plt.xlabel('frequency (%s)' % (freqfactor[1]))
                plt.ylabel('p-value')
                plt.semilogx(freqs, pvalue, label='pvalue', linestyle='--')
                plt.title(data_name + ': Fourier power distributions, Shapiro-Wilks normality test')
                plt.legend(loc=3, framealpha=0.3, fontsize=10)
                plt.savefig(savefig + '.shapiro_wilks.distribution_width.%s' % (savefig_format))

                ###############################################################
                # Plot the results of the Anderson-Darling test for the Fourier
                # power distributions.  If the AD statistic value is above a
                # particular critical value corresponding to a given
                # significance level, then the null hypothesis (in this case
                # normality) can be rejected at that significance level.
                statvalue = np.asarray([result[0] for result in anderson_darling])
                crit = anderson_darling[0][1]
                significance = anderson_darling[0][2]
                plt.figure(15)
                plt.xlabel('frequency (%s)' % (freqfactor[1]))
                plt.ylabel('Anderson-Darling statistic value')
                for jjj, sig in enumerate(significance):
                    plt.axhline(crit[jjj], color='k')
                    plt.text(freqs[1], crit[jjj], '%s' % (str(np.rint(sig))))
                plt.semilogx(freqs, statvalue, label='AD statistic', linestyle='--')
                plt.title(data_name + ': Fourier power distributions, Anderson-Darling normality test')
                plt.legend(loc=3, framealpha=0.3, fontsize=10)
                plt.savefig(savefig + '.anderson_darling.distribution_width.%s' % (savefig_format))

                ###############################################################
                # Save various data products
                # Fourier Power of the analyzed data
                ofilename = region_id
                pkl_write(pkl_location,
                          'OUT.' + ofilename + '.fourier_power.pickle',
                          (freqs / freqfactor[0], pwr))

                # Analyzed data
                pkl_write(pkl_location,
                          'OUT.' + ofilename + '.dc_analysed.pickle',
                          (t, dc_analysed))

                # Fourier transform
                pkl_write(pkl_location,
                          'OUT.' + ofilename + '.fft_transform.pickle',
                          (freqs / freqfactor[0], fft_transform))

                # Arithmetic mean of power spectra
                pkl_write(pkl_location,
                          'OUT.' + ofilename + '.iobs.pickle',
                          (freqs / freqfactor[0], np.log(iobs)))

                # Geometric mean of power spectra
                #(freqs / freqfactor[0], logiobs, logiobs_distrib_width))
                pkl_write(pkl_location,
                          'OUT.' + ofilename + '.logiobs.pickle',
                          (freqs / freqfactor[0], logiobs, iobs_peak, logiobs_peak_location, nx * ny, ccc0, lag, ccclag))

                # Widths of the power distributions
                pkl_write(pkl_location,
                          'OUT.' + ofilename + '.distribution_widths.pickle',
                          (freqs / freqfactor[0], logiobs_distrib_width, logiobs_std, np.abs(logiobs - logiobs_peak_location)))

                # Error in the Widths of the power distributions
                pkl_write(pkl_location,
                          'OUT.' + ofilename + '.distribution_widths_error.pickle',
                          (freqs / freqfactor[0], error_logiobs_distrib_width))

                # Average coherence
                pkl_write(pkl_location,
                          'OUT.' + ofilename + '.coherence.pickle',
                          (freqs / freqfactor[0], coher, coher_max))

                # Bump fit
                #pkl_write(pkl_location,
                #          'OUT.' + ofilename + '.bump_fit_all.pickle',
                #          (bump_ans_all, bump_err_all))

                # Simple fit
                #pkl_write(pkl_location,
                #          'OUT.' + ofilename + '.simple_fit_all.pickle',
                #          (simple_ans_all, simple_err_all))

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
          corename='shutdownfun6_6hr',
          sunlocation='limb',
          fits_level='1.0',
          waves=['171'],
          regions=['highlimb', 'crosslimb', 'lowlimb', 'moss', 'loopfootpoints1', 'loopfootpoints2'],
          windows=['hanning'],
          manip='relative')
"""

"""
do_lstsqr(dataroot='~/Data/AIA/',
          ldirroot='~/ts/pickle_cc/',
          sfigroot='~/ts/img_cc/',
          scsvroot='~/ts/csv_cc/',
          corename='shutdownfun3_1hr',
          sunlocation='disk',
          fits_level='1.5',
          waves=['171'],
          regions=['moss', 'sunspot', 'qs', 'loopfootpoints'],
          windows=['hanning'],
          manip='relative')
"""


do_lstsqr(dataroot='~/Data/AIA/',
          ldirroot='~/ts/pickle_cc_final/',
          sfigroot='~/ts/img_cc_final/',
          scsvroot='~/ts/csv_cc_final/',
          corename='shutdownfun3_6hr',
          sunlocation='disk',
          fits_level='1.5',
          waves=['171', '193'],
          regions=['qs', 'moss', 'loopfootpoints', 'sunspot'],
          windows=['hanning'],
          manip='relative')

"""
do_lstsqr(dataroot='~/Data/AIA/',
          ldirroot='~/ts/pickle',
          sfigroot='~/ts/img/',
          scsvroot='~/ts/csv/',
          corename='20120923_0000__20120923_0100',
          sunlocation='disk',
          fits_level='1.5',
          waves=['171'],
          regions=['moss', 'sunspot', 'loopfootpoints', 'qs'],
          windows=['hanning'],
          manip='relative',
          savefig_format='png')
"""