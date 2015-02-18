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
import datatools
from py.OLD import aia_plaw
from py.aia.OLD.paper1 import log_10_product, tsDetails, s3min, s5min, s_U68, s_U95, s_L68, s_L95
from py.aia.OLD.paper1 import prettyprint, csv_timeseries_write, pkl_write
from py.aia.OLD.paper1 import descriptive_stats
import scipy
from scipy.interpolate import interp1d
plt.ioff()


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


# Main analysis loop
dataroot = '~/Data/AIA/'
ldirroot = '~/ts/pickle_cc_False_dr_False/'
sfigroot = '~/ts/img_cc_False_dr_False/'
scsvroot = '~/ts/csv_cc_False_dr_False/'
corename = 'study2'
#sunlocation = 'equatorial'
sunlocation = 'spoca665'
fits_level = '1.5'

# Wavelengths we want to analyze
waves = ['171', '193']

# Regions we are interested in
regions = ['R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7']

# Apodization windows
windows = ['hanning']

# Output images in what file format?
savefig_format = 'png'



# main loop
for iwave, wave in enumerate(waves):

    for iregion, region in enumerate(regions):
        # Create the branches in order
        branches = [corename, sunlocation, fits_level, wave, region]

        # Set up the roots we are interested in
        roots = {"pickle": ldirroot,
                 "image": sfigroot,
                 "csv": scsvroot}

        # Data and save locations are based here
        locations = datatools.save_location_calculator(roots,
                                     branches)

        # set the saving locations
        sfig = locations["image"]
        scsv = locations["csv"]

        # Identifier
        ident = datatools.ident_creator(branches)

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
            region_id = '.'.join((ident, window))

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

            # Create a location to save the figures
            savefig = os.path.join(os.path.expanduser(sfig), window, manip)
            if not(os.path.isdir(savefig)):
                os.makedirs(savefig)
            savefig = os.path.join(savefig, region_id)

            # Create a time series object
            t = dt * np.arange(0, nt)
            tsdummy = TimeSeries(t, t)
            freqs_original = tsdummy.PowerSpectrum.frequencies.positive
            posindex = np.fft.fftfreq(nt, dt) > 0.0 #tsdummy.PowerSpectrum.frequencies.posindex
            iobs = np.zeros(tsdummy.PowerSpectrum.ppower.shape)
            logiobs = np.zeros(tsdummy.PowerSpectrum.ppower.shape)
            logiobsDC = np.zeros(tsdummy.PowerSpectrum.ppower.shape)
            nposfreq = len(iobs)
            nfreq = tsdummy.PowerSpectrum.frequencies.nfreq

            # storage
            pwr = np.zeros((ny, nx, nposfreq))
            dtotal = np.zeros((ny, nx))
            dmax = np.zeros_like(dtotal)
            dmin = np.zeros_like(dtotal)
            dsd = np.zeros_like(dtotal)
            dlnsd = np.zeros_like(dtotal)

            for i in range(0, nx):
                for j in range(0, ny):

                    # Get the next time-series
                    d = dc[j, i, :].flatten()

                    # Fix the data for any non-finite entries
                    d = tsutils.fix_nonfinite(d)

                    # Get the total emission
                    dtotal[j, i] = np.sum(d)

                    # Get the maximum emission
                    dmax[j, i] = np.max(d)

                    # Get the minimum emission
                    dmin[j, i] = np.min(d)

                    # Get the standard deviation of the emission
                    dsd[j, i] = np.std(d)

                    # Get the standard deviation of the log of the emission
                    dlnsd[j, i] = np.std(np.log(d))

                    # Multiply the data by the apodization window
                    d = ts_apply_window(d, win)

                    # Define the Fourier power we are analyzing
                    this_power = (np.abs(np.fft.fft(d)) ** 2)[posindex] / (1.0 * nt)#ts.PowerSpectrum.ppower

                    # Store the individual Fourier power
                    pwr[j, i, :] = this_power



            # Save the Fourier Power of the analyzed time-series
            ofilename = region_id
            pkl_write(pkl_location,
                      'OUT.' + ofilename + '.fourier_power.pickle',
                      (freqs_original, pwr))




