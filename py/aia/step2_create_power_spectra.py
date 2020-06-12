"""
Step 2
Create power spectra and other summary statistics of the time-series at each
pixel in each region.
"""
import os
import argparse
from scipy.interpolate import interp1d
import numpy as np
import astropy.units as u

from tools import tsutils
from tools.tstools import is_evenly_sampled
from tools.timeseries import TimeSeries
from tools.pstools import create_simulated_power_spectra2
import details_study as ds


parser = argparse.ArgumentParser(description='Create power spectra for the results from one or more channels.')
parser.add_argument('-w', '--waves', help='comma separated list of channels', type=str)
parser.add_argument('-s', '--study', help='comma separated list of study types', type=str)
args = parser.parse_args()
waves = [item for item in args.waves.split(',')]
study_types = [item for item in args.study.split(',')]


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
    return win


# Apodization windows
window = 'hanning'

# Absolute tolerance in seconds when deciding if data is evenly sampled
absolute_tolerance = ds.absolute_tolerance

#
# Main FFT power and time-series characteristic preparation loop
#
for study_type in study_types:
    for iwave, wave in enumerate(waves):
        # General notification that we have a new data-set
        print('\nLoading New Data')

        # branch location
        b = [study_type, ds.original_datatype, wave]

        # Location of the project data
        directory = ds.datalocationtools.save_location_calculator(ds.roots, b)["project_data"]

        # Input filename
        input_filename = '{:s}_{:s}.step1.npz'.format(study_type, wave)

        # Input filepath
        input_filepath = os.path.join(directory, input_filename)

        # Load the time series data
        if ds.use_time_series_data:
            print('Loading ' + input_filepath)
            dc = np.load(input_filepath)['arr_0']
            t = np.load(input_filepath)['arr_1']

            # Get some properties of the datacube
            ny = dc.shape[0]
            nx = dc.shape[1]
            nt = dc.shape[2]
            # Should be evenly sampled data.  It not, then resample to get
            # evenly sampled data
            dt = ds.target_cadence
            ts_evenly_sampled = is_evenly_sampled(t, absolute_tolerance.to('s').value)
            if not ts_evenly_sampled:
                print('Resampling data to even cadence.')
                dt = (t[-1] - t[0]) / (1.0 * (nt - 1))
                print('Resampling to an even time cadence of {:n} seconds'.format(dt))
                evenly_sampled_t = np.arange(0, nt) * dt
                for iii in range(0, ny):
                    for jjj in range(0, nx):
                        f = interp1d(t, dc[iii, jjj, :])
                        dc[iii, jjj, :] = f(evenly_sampled_t)
                        t = evenly_sampled_t
            else:
                print('Evenly sampled to within tolerance.')

            # calculate a window function
            win = DefineWindow(window, nt)

            # Create a dummy time series object to get some frequency
            # information
            t = dt * np.arange(0, nt)
            tsdummy = TimeSeries(t * u.s, t)

            # Positive frequencies
            pfrequencies = tsdummy.FFTPowerSpectrum.frequencies.pfrequencies

            # The index of the positive frequencies
            pindex = tsdummy.FFTPowerSpectrum.frequencies.pindex

        else:
            # Create the simulated power spectra
            nx = ds.nx
            ny = ds.ny
            pfrequencies = ds.pfrequencies
            true_parameters = [ds.amplitude, int(wave)/100, ds.white_noise]
            simulated_power_spectra = create_simulated_power_spectra2(nx,
                                                                      ny,
                                                                      ds.simulation_model,
                                                                      true_parameters, pfrequencies)

        # Number of positive frequencies
        nposfreq = len(pfrequencies)

        # Storage - Fourier power
        pwr = np.zeros((ny, nx, nposfreq))

        # Storage - Fourier power
        pwr_rel = np.zeros_like(pwr)

        # Storage - sum of log of relative intensity
        drel_power = np.zeros(nposfreq)

        # Storage - summary stats
        dtotal = np.zeros((ny, nx))
        dmax = np.zeros_like(dtotal)
        dmin = np.zeros_like(dtotal)
        dsd = np.zeros_like(dtotal)
        dlnsd = np.zeros_like(dtotal)

        for i in range(0, nx):
            for j in range(0, ny):

                if ds.use_time_series_data:
                    # Get the next time-series
                    d = dc[j, i, :].flatten()

                    # Fix the data for any non-finite entries
                    d = tsutils.fix_nonfinite(d)

                    # Fourier transform of the absolute intensities
                    # Multiply the data by the apodization window
                    d_with_window = apply_window(d, win)

                    # Get the Fourier transform
                    this_fft = np.fft.fft(d_with_window)

                    # Fourier power of the absolute intensities
                    this_power = ((np.abs(this_fft) ** 2) / (1.0 * nt))[pindex]

                    # Relative change in intensity
                    #dmean = np.mean(d)
                    #d_relative_change = (d - dmean) / dmean
                    #d_relative_change_with_window = apply_window(d_relative_change, win)
                    #this_fft_relative_change_with_window = np.fft.fft(d_relative_change_with_window)
                    #this_power_relative_change_with_window = ((np.abs(this_fft_relative_change_with_window) ** 2) / (1.0 * nt))[pindex]
                    #pwr_rel[j, i, :] = this_power_relative_change_with_window[:]
                    # Ireland et al (2015) summation
                    # Sum over the log(Fourier power of the relative intensities)
                    #drel_power += np.log10(this_power_relative_change_with_window)
                else:
                    this_power = simulated_power_spectra[i, j, :]

                # Store the individual Fourier power
                pwr[j, i, :] = this_power

        # Save the Fourier power of the absolute intensities
        output_filename = '{:s}_{:s}_{:s}.absolute.step2.npz'.format(study_type, wave, window)
        output_filepath = os.path.join(directory, output_filename)
        print('Saving power spectra to ' + output_filepath)
        np.savez(output_filepath, pwr, pfrequencies)

        """
        # Save the Fourier power of the relative intensities
        output_filename = '{:s}_{:s}_{:s}.relative_change.step2.npz'.format(ds.study_type, wave, window)
        output_filepath = os.path.join(directory, output_filename)
        print('Saving power spectra to ' + output_filepath)
        np.savez(output_filepath, pwr_rel, pfrequencies)
    
        # Save the sum over the log(Fourier power of the relative intensities).  This is the kind of
        # summing done in Ireland et al (2015).
        output_filename = '{:s}_{:s}_{:s}.sum_log10_relative_change.step2.npz'.format(ds.study_type, wave, window)
        output_filepath = os.path.join(directory, output_filename)
        print('Saving power spectrum to ' + output_filepath)
        np.savez(output_filepath, drel_power, pfrequencies)
    
        # Save the sum of the Fourier power of the absolute intensities
        output_filename = '{:s}_{:s}_{:s}.sum_absolute.step2.npz'.format(ds.study_type, wave, window)
        output_filepath = os.path.join(directory, output_filename)
        print('Saving power spectrum to ' + output_filepath)
        np.savez(output_filepath, np.sum(pwr, axis=(0, 1)), pfrequencies)
    
        # Save the sum of the log(Fourier power of the absolute intensities)
        output_filename = '{:s}_{:s}_{:s}.sum_log10_absolute.step2.npz'.format(ds.study_type, wave, window)
        output_filepath = os.path.join(directory, output_filename)
        print('Saving power spectrum to ' + output_filepath)
        np.savez(output_filepath, np.sum(np.log10(pwr), axis=(0, 1)), pfrequencies)
        """
