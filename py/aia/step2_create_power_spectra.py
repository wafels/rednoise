"""
Step 2
Create power spectra and other summary statistics of the time-series at each
pixel in each region.
"""
import os

import numpy as np
import tsutils
from timeseries import TimeSeries

import cPickle as pickle
from scipy.interpolate import interp1d

from tstools import is_evenly_sampled
from timeseries import TimeSeries

import study_details as sd


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


# Wavelengths we want to analyze
waves = ['171', '193']

# Regions we are interested in
regions = ['R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7']

# Apodization windows
windows = ['hanning']

# Output images in what file format?
savefig_format = 'png'

#
# Main FFT power and time-series characteristic preparation loop
#
for iwave, wave in enumerate(waves):

    for iregion, region in enumerate(regions):
        # Create the branches in order
        branches = [sd.corename, sd.sunlocation, sd.fits_level, sd.wave, region]

        # Set up the roots we are interested in
        roots = {"pickle": sd.ldirroot,
                 "image": sd.sfigroot,
                 "csv": sd.scsvroot}

        # Data and save locations are based here
        locations = sd.save_location_calculator(roots, branches)

        # set the saving locations
        sfig = locations["image"]
        scsv = locations["csv"]

        # Identifier
        ident = sd.ident_creator(branches)

        # Go through all the windows
        for iwindow, window in enumerate(windows):
            # General notification that we have a new data-set
            print('Loading New Data')
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

            # Should be evenly sampled data.  It not, then resample to get
            # evenly sampled data
            t = time_information["time_in_seconds"]
            dt = 12.0
            ts_evenly_sampled = is_evenly_sampled(t, expected=dt, absolute=dt / 100.0)
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

            # Create a dummy time series object
            t = dt * np.arange(0, nt)
            tsdummy = TimeSeries(t, t)

            # Positive frequencies
            pfrequencies = tsdummy.FFTPowerSpectrum.frequencies.pfrequencies

            # The index of the positive frequencies
            pindex = tsdummy.FFTPowerSpectrum.frequencies.pindex

            # Number of positive frequencies
            nposfreq = len(pfrequencies)

            # Storage - Fourier power
            pwr = np.zeros((ny, nx, nposfreq))

            # Storage - summary stats
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

                    # Define the time series
                    ts = TimeSeries(t, d)

                    # Get the Fourier power we are analyzing
                    this_power = ts.FFTPowerSpectrum.ppower

                    # Store the individual Fourier power
                    pwr[j, i, :] = this_power

            # Save the Fourier Power of the analyzed time-series
            ofilename = 'OUT.' + region_id
            pkl_write(pkl_location,
                      ofilename + '.fourier_power.pickle',
                      (pfrequencies, pwr))

            # Save the summary statistics
            np.savez(ofilename + '.summary_stats.npy',
                     dtotal, dmax, dmin, dsd, dlnsd)
