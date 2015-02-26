"""
Step 2
Create power spectra and other summary statistics of the time-series at each
pixel in each region.
"""
import os
import csv

import numpy as np
import tsutils
from timeseries import TimeSeries

import cPickle as pickle
from scipy.interpolate import interp1d

import astropy.units as u

from tstools import is_evenly_sampled
from timeseries import TimeSeries

import study_details as sd


# Apply the window
def apply_window(d, win):
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
regions = ['sunspot', 'loop footpoints', 'quiet Sun', 'moss']

# Apodization windows
windows = ['hanning']

# Absolute tolerance in seconds when deciding if data is evenly sampled
absolute_tolerance = sd.absolute_tolerance

#
# Main FFT power and time-series characteristic preparation loop
#
for iwave, wave in enumerate(waves):

    for iregion, region in enumerate(regions):

        # branch location
        b = [sd.corename, sd.sunlocation, sd.fits_level, wave, region]

        # Region identifier name
        region_id = sd.datalocationtools.ident_creator(b)

        # Output location
        output = sd.datalocationtools.save_location_calculator(sd.roots, b)["pickle"]

        # Output filename
        ofilename = os.path.join(output, region_id + '.datacube')

        # Go through all the windows
        for iwindow, window in enumerate(windows):
            # General notification that we have a new data-set
            print('\nLoading New Data')
            # Which wavelength?
            print('Wave: ' + wave + ' (%i out of %i)' % (iwave + 1, len(waves)))
            # Which region
            print('Region: ' + region + ' (%i out of %i)' % (iregion + 1, len(regions)))
            # Which window
            print('Window: ' + window + ' (%i out of %i)' % (iwindow + 1, len(windows)))

            # Load the data
            pkl_file_location = os.path.join(output, ofilename + '.pkl')
            print('Loading ' + pkl_file_location)
            pkl_file = open(pkl_file_location, 'rb')
            dc = pickle.load(pkl_file)
            time_information = pickle.load(pkl_file)
            pkl_file.close()

            # Get some properties of the datacube
            ny = dc.shape[0]
            nx = dc.shape[1]
            nt = dc.shape[2]

            # Should be evenly sampled data.  It not, then resample to get
            # evenly sampled data
            t = time_information["time_in_seconds"]
            dt = sd.target_cadence
            ts_evenly_sampled = is_evenly_sampled(t,
                                                  absolute_tolerance.to('s').value)
            if not is_evenly_sampled:
                print('Resampling to an even time cadence.')
                dt = (t[-1] - t[0]) / (1.0 * (nt - 1))
                evenly_sampled_t = np.arange(0, nt) * dt
                for iii in range(0, nx):
                    for jjj in range(0, ny):
                        f = interp1d(t, dc[iii, jjj, :])
                        dc[iii, jjj, :] = f(evenly_sampled_t)
                t = evenly_sampled_t
            else:
                print('Evenly sampled to within tolerance.')

            # Define an array to store the analyzed data
            dc_analysed = np.zeros_like(dc)

            # calculate a window function
            win, dummy_winname = DefineWindow(window, nt)

            # Create a dummy time series object to get some frequency
            # information
            t = dt * np.arange(0, nt)
            tsdummy = TimeSeries(t * u.s, t)

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

            logiobs = np.zeros(nposfreq)
            for i in range(0, nx):
                for j in range(0, ny):

                    # Get the next time-series
                    d = dc[j, i, :].flatten()

                    # Fix the data for any non-finite entries
                    d = tsutils.fix_nonfinite(d)

                    # Relative change in intensity
                    dmean = np.mean(d)
                    d = (d - dmean) / dmean

                    # Multiply the data by the apodization window
                    d = apply_window(d, win)

                    # Get the Fourier power we will analyze later
                    this_power = ((np.abs(np.fft.fft(d)) ** 2) / (1.0 * nt))[pindex]

                    # Store the individual Fourier power
                    pwr[j, i, :] = this_power

                    # Summed logarithm of the power
                    logiobs = logiobs + np.log(this_power)

            # Save the Fourier Power of the analyzed time-series
            logiobs = logiobs / np.float(nx * ny)
            ofilename = ofilename + '.' + window
            filepath = os.path.join(output, ofilename + '.fourier_power_for_gelu.csv')
            print('Saving power spectra to ' + filepath)
            with open(filepath, 'wb') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',')
                spamwriter.writerow(['frequencies (Hz)', 'sum_over_region(log(fourier power))'])
                for i in range(0, nposfreq):
                    spamwriter.writerow([str(pfrequencies[i].value), str(logiobs[i])])

            filepath = os.path.join(output, ofilename + '.fourier_power_for_gelu_number_of_spectra.csv')
            print('Saving power spectra to ' + filepath)
            with open(filepath, 'wb') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',')
                spamwriter.writerow(['number of spectra = ', str(nx * ny)])

