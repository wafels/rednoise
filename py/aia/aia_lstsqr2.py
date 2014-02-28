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
#from rnsimulation import SimplePowerLawSpectrumWithConstantBackground, TimeSeriesFromPowerSpectrum
#from rnfit2 import Do_MCMC, rnsave
#from pymcmodels import single_power_law_with_constant_not_normalized
import csv
import cPickle as pickle
import aia_specific
import aia_plaw

font = {'family': 'normal',
        'weight': 'bold',
        'size': 12}

matplotlib.rc('font', **font)
matplotlib.rc('lines', linewidth=1)
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

def log_10_product(x, pos):
    """The two args are the value and tick position.
    Label ticks with the product of the exponentiation"""
    return '%5.1f' % (x)

# String defining the basics number of time series
def tsDetails(nx, ny, nt):
    return '[%i t.s., %i samples]' % (nx * ny, nt)


# Apply the manipulation function
def ts_manip(d, manip):
    if manip == 'relative':
        dmean = np.mean(d)
        d = (d - dmean) / dmean
    return d


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
              savefig_format='eps',
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
                    data_name = 'AIA ' + str(wave) + ' : ' + sunday_name[region]
                else:
                    data_name = 'AIA ' + str(wave) + ' : ' + region

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
                iobs = np.zeros(tsdummy.PowerSpectrum.Npower.shape)
                logiobs = np.zeros(tsdummy.PowerSpectrum.Npower.shape)
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
                        this_power = ts.PowerSpectrum.ppower

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

                """
                # Set up the plot with the x axis playing nice
                ax = plt.subplot(111)
                ax.set_xscale('log')
                ax.set_yscale('log')
                xformatter = plt.FuncFormatter(log_10_product)
                ax.xaxis.set_major_formatter(xformatter)

                ax.plot(freqs, iobs)
                plt.xlabel('frequency (%s)' % (freqfactor[1]))
                plt.savefig(savefig + '.data.png')
                """
                ###############################################################
                # Power spectrum analysis: trying to fit a spectrum with a bump
                # -------------------------------------------------------------
                # Trying to fit the data with a bump.  We can use the answers
                # found with a fit without a bump, as an initial estimate
                #
                # Without the bump: power = kf ** -n + c
                #
                # loglog plots of the power seem to show a bump
                #
                # log(power) = log( kf ** -n + c ) + GaussianShape
                #
                # Note that
                #
                # log(power/ (kf ** -n + c)) = GaussianShape
                #
                # This suggests a two stage fitting process.  First, fit the
                # background power spectrum.  Secondly, fit the Gaussian shape
                #
                # Also,
                #
                # power = (kf ** -n + c) * exp(GaussianShape)
                #
                # What does this mean physically?
                #
                # A better fit is probably obtainable by fitting both the power law and
                # the Gaussian shape simultaneously.
                #
                """
                bf_diff = iobs / bf
                plt.loglog(freqs, bf_diff)
                ga_est, ga_est_error = aia_plaw.do_fit(np.log(x),
                                                       np.log(bf_diff),
                                                       aia_plaw.GaussianShape, p0 = [1.0, 1.9, 0.6])
                GA = aia_plaw.GaussianShape(np.log(x), ga_est[0, 0, 0], ga_est[0, 0, 1], ga_est[0, 0, 2])
                g_A = ga_est[0, 0, 0]
                g_c = 1000 * np.exp(ga_est[0, 0, 1]) * xnorm
                g_s = ga_est[0, 0, 2] / np.log(10.0)

                # Plot of the Gaussian excess
                plt.figure(1)
                plt.loglog(freqs, bf_diff, color='b', label='data / (power law fit)')
                plt.loglog(freqs, np.exp(GA), color='r', label='Gaussian fit: A=%4.2f, $f_{0}=$%4.2f mHz, $log_{10}\sigma=$%4.2f' % (g_A, g_c, g_s))
                plt.xlabel('frequency (%s)' % (freqfactor[1]))
                plt.ylabel('power ratio')
                plt.title(data_name + ' - arithmetic mean')
                plt.legend(loc=3, fontsize=10, framealpha=0.5)
                plt.savefig(savefig + '.aPs_gauss_estimate.png')
                gwb, gwb_error = aia_plaw.do_fit(x, np.log(iobs),
                                      aia_plaw.LogPowerLawPlusConstantGaussian, nvar= 6,p0 = [answer[0, 0, 0],
                                                   answer[0, 0, 1],
                                                   answer[0, 0, 2],
                                                   ga_est[0,0,0], ga_est[0,0,1], ga_est[0,0,2]],)

                bf_gwb = aia_plaw.LogPowerLawPlusConstantGaussian(x, gwb[0,0,0], gwb[0,0,1], gwb[0,0,2], gwb[0,0,3], gwb[0,0,4], gwb[0,0,5])

                # Plot of the joint fit
                plt.figure(2)
                plt.loglog(freqs, bf * np.exp(GA), color='g', label='fit separately')
                plt.loglog(freqs, iobs, color='b', label='data')
                plt.loglog(freqs, np.exp(bf_gwb), color='r', label='fit together')
                plt.legend(loc=3, fontsize=10, framealpha=0.5)
                g_A = gwb[0, 0, 3]
                g_c = 1000 * np.exp(gwb[0, 0, 4]) * xnorm
                g_s = gwb[0, 0, 5] / np.log(10.0)
                label_gaussian = 'FT: Gaussian fit: A=%4.2f, $f_{0}=$%4.2f mHz, $log_{10}\sigma=$%4.2f' % (g_A, g_c, g_s)
                label_pwrlaw = 'FT: power law index = %4.2f' % gwb[0, 0, 1]
                plt.text(freqs[0], 1.0, label_gaussian)
                plt.text(freqs[0], 0.1, label_pwrlaw)
                plt.title(data_name + ' - arithmetic mean')
                plt.savefig(savefig + '.aPs_best_fit_with_Gaussian.png')
                plt.close('all')
                print answer, ga_est
                print gwb

                #sdf = aflakhflahdf
                """

                # Fourier power: get a Time series from the arithmetic sum of
                # all the time-series at every pixel, then apply the
                # manipulation and the window. Find the Fourier power
                # and do the fit.
                doriginal_ts_iobs = doriginal_ts.PowerSpectrum.ppower
                answer_doriginal_ts = curve_fit(aia_plaw.LogPowerLawPlusConstant, x, np.log(doriginal_ts_iobs), p0=answer[0])
                param_dts = answer_doriginal_ts[0]
                bf_dts = np.exp(aia_plaw.LogPowerLawPlusConstant(x, param_dts[0], param_dts[1], param_dts[2]))
                nerr_dts = np.sqrt(answer_doriginal_ts[1][1, 1])



                ax = plt.subplot(111)
                ax.set_xscale('log')
                ax.set_yscale('log')
                xformatter = plt.FuncFormatter(log_10_product)
                ax.xaxis.set_major_formatter(xformatter)
                
                ax.plot(freqs, doriginal_ts_iobs, color='r', label='sum over region')
                ax.plot(freqs, bf_dts, color='r', linestyle="--", label='fit to sum over region n=%4.2f +/- %4.2f' % (param_dts[1], nerr_dts))

                # Arithmetic mean of the power spectra from each pixel
                ax.plot(freqs, iobs, color='b', label='arithmetic mean of power spectra from each pixel (Erlang distributed)')
                ax.plot(freqs, bf, color='b', linestyle="--", label='fit to arithmetic mean of power spectra from each pixel n=%4.2f +/- %4.2f' % (param[1], nerr))
                #plt.loglog(freqs, bf_gwb, color='b', linestyle='-.', label = 'bf_gwb')

                # Extra information for the plot
                ax.axvline(five_min, color='k', linestyle='-.', label='5 mins.')
                ax.axvline(three_min, color='k', linestyle='--', label='3 mins.')
                #plt.axhline(1.0, color='k', label='average power')
                plt.xlabel('frequency (%s)' % (freqfactor[1]))
                plt.ylabel('normalized power [%i time series, %i samples each]' % (nx * ny, nt))
                plt.title(data_name + ' - arithmetic mean')
                plt.grid()
                plt.legend(loc=3, fontsize=10, framealpha=0.5)
                #plt.text(freqs[0], 1.0, 'note: least-squares fit used, but data is not Gaussian distributed', fontsize=8)
                plt.savefig(savefig + '.arithmetic_mean_power_spectra.%s' % (savefig_format))
                

                """
                # Plots of power spectra: arithmetic means of summed emission
                # and summed power spectra
                plt.figure(1)

                # Arithmetic mean of all the time series, then analysis
                plt.loglog(freqs, doriginal_ts_iobs, color='r', label='sum over region')
                plt.loglog(freqs, bf_dts, color='r', linestyle="--", label='fit to sum over region n=%4.2f +/- %4.2f' % (param_dts[1], nerr_dts))

                # Arithmetic mean of the power spectra from each pixel
                plt.loglog(freqs, iobs, color='b', label='arithmetic mean of power spectra from each pixel (Erlang distributed)')
                plt.loglog(freqs, bf, color='b', linestyle="--", label='fit to arithmetic mean of power spectra from each pixel n=%4.2f +/- %4.2f' % (param[1], nerr))
                #plt.loglog(freqs, bf_gwb, color='b', linestyle='-.', label = 'bf_gwb')

                # Extra information for the plot
                plt.axvline(five_min, color='k', linestyle='-.', label='5 mins.')
                plt.axvline(three_min, color='k', linestyle='--', label='3 mins.')
                #plt.axhline(1.0, color='k', label='average power')
                plt.xlabel('frequency (%s)' % (freqfactor[1]))
                plt.ylabel('normalized power [%i time series, %i samples each]' % (nx * ny, nt))
                plt.title(data_name + ' - arithmetic mean')
                plt.legend(loc=3, fontsize=10, framealpha=0.5)
                #plt.text(freqs[0], 1.0, 'note: least-squares fit used, but data is not Gaussian distributed', fontsize=8)
                plt.savefig(savefig + '.arithmetic_mean_power_spectra.%s' % (savefig_format))
                """
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

                # Geometric mean of power spectra at each pixel
                plt.figure(2)
                plt.loglog(freqs, np.exp(logiobs), label='geometric mean of power spectra at each pixel')
                plt.loglog(freqs, bf2, color='k', label='best fit n=%4.2f +/- %4.2f' % (param2[1], nerr2))
                plt.loglog(freqs, lim[0, 0, :], linestyle='--', label='lower 68%')
                plt.loglog(freqs, lim[0, 1, :], linestyle='--', label='upper 68%')
                plt.loglog(freqs, lim[1, 0, :], linestyle=':', label='lower 95%')
                plt.loglog(freqs, lim[1, 1, :], linestyle=':', label='upper 95%')
                plt.axvline(five_min, color='k', linestyle='-.', label='5 mins.')
                plt.axvline(three_min, color='k', linestyle='--', label='3 mins.')
                plt.xlabel('frequency (%s)' % (freqfactor[1]))

                plt.ylabel('power [%i time series, %i samples each]' % (nx * ny, nt))
                plt.title(data_name + ' - geometric mean')
                plt.legend(loc=3, fontsize=10, framealpha=0.5)
                plt.savefig(savefig + '.geometric_mean_power_spectra.%s' % (savefig_format))
                # plot some histograms of the log power at a small number of
                # frequencies.
                findex = []
                f_of_interest = [0.5 * five_min, five_min, three_min, 2 * three_min, 3 * three_min]
                for thisf in f_of_interest:
                    findex.append(np.unravel_index(np.argmin(np.abs(thisf - freqs)), freqs.shape)[0])
                plt.figure(3)
                plt.xlabel('$\log_{10}(power)$')
                plt.ylabel('proportion found at given frequency')
                plt.title(data_name + ' - power distributions')
                for f in findex:
                    plt.plot(h[1][1:] / np.log(10.0), hpwr[f, :], label='%7.2f %s' % (freqs[f], freqfactor[1]))
                plt.legend(loc=3, fontsize=10, framealpha=0.5)
                plt.savefig(savefig + '.power_spectra_distributions.%s' % (savefig_format))

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
                plt.axvline(five_min, color='k', linestyle='-.', label='5 mins.')
                plt.axvline(three_min, color='k', linestyle='--', label='3 mins.')
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
                # Save various data products
                # Fourier Power of the analyzed data
                ofilename = region_id
                pkl_write(pkl_location,
                          'OUT.' + ofilename + '.fourier_power.pickle',
                          (freqs, pwr))

                # Analyzed data
                pkl_write(pkl_location,
                          'OUT.' + ofilename + '.dc_analysed.pickle',
                          (t, dc_analysed))

                # Fourier transform
                pkl_write(pkl_location,
                          'OUT.' + ofilename + '.fft_transform.pickle',
                          (freqs, fft_transform))

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
          waves=['171', '193', '211', '131'],
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
          regions=['sunspot', 'qs', 'loopfootpoints', 'moss'],
          windows=['hanning'],
          manip='relative')
"""
"""
do_lstsqr(dataroot='~/Data/AIA/',
          ldirroot='~/ts/pickle_cc/',
          sfigroot='~/ts/img_cc/',
          scsvroot='~/ts/csv_cc/',
          corename='shutdownfun3_6hr',
          sunlocation='disk',
          fits_level='1.5',
          waves=['193'],
          regions=['moss', 'sunspot', 'loopfootpoints', 'qs'],
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
          regions=['moss', 'sunspot', 'qs', 'loopfootpoints'],
          windows=['hanning'],
          manip='relative',
          savefig_format='png')




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