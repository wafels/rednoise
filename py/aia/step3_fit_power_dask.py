#
# Step 3.  Load in the FFT power spectra and fit models. Save the results
#
import os
import argparse
import numpy as np

# import distributed
from dask.distributed import Client, LocalCluster

from stingray import Powerspectrum
from stingray.modeling import PSDLogLikelihood
from stingray.modeling import PSDParEst

import astropy.units as u
from astropy.time import Time
from spectral_model_parameter_estimators import InitialParameterEstimatePlC, OldSchoolInitialParameterEstimatePlC
from details_spectral_models import SelectModel
import details_study as ds


parser = argparse.ArgumentParser(description='Fit a model to Fourier power spectra (from step2).')
parser.add_argument('-w', '--waves', help='comma separated list of channels', type=str)
args = parser.parse_args()

parser = argparse.ArgumentParser(description='Which study type(s) to load.')
parser.add_argument('-s', '--study', help='comma separated list of study types', type=str)
args = parser.parse_args()

# Data to analyze
# Wavelengths we want to analyze
waves = [item for item in args.waves.split(',')]

# Studies to load in
study_types = [item for item in args.study.split(',')]

# Type of power spectrum
power_type = 'absolute'

# Window used
window = 'hanning'

# Analyze a subsection of the data?
analyze_subsection = False
if analyze_subsection:
    side = 8
    subsection = ((128-side, 128+side), (128-side, 128+side))
else:
    subsection = ((0, None), (0, None))

# Normalize the frequencies?
normalize_frequencies = True

# Divide by the initial power?
divide_by_initial_power = True

# Define the observation model
this_model = SelectModel('pl_c')
observation_model = this_model.observation_model
scipy_optimize_options = this_model.scipy_optimize_options


def load_spectrum(directory, study_type, wave, window, power_type):
    filename = '{:s}_{:s}_{:s}.{:s}.step2.npz'.format(study_type, wave, window, power_type)
    filepath = os.path.join(directory, filename)
    print('Loading power spectra from ' + filepath)
    return np.load(filepath)


def dask_fit_fourier_pl_c(power_spectrum):
    """
    Fits the power law + constant observation model

    Parameters
    ----------
    power_spectrum : ~tuple
        The first entry corresponds to the positive frequencies of the power spectrum.
        The second entry corresponds to the Fourier power at those frequencies

    Return
    ------
    A parameter estimation object - see the Stingray docs.

    """
    # Make a Powerspectrum object
    ps = Powerspectrum()
    ps.freq = power_spectrum[0]
    ps.power = power_spectrum[1]
    ps.df = ps.freq[1] - ps.freq[0]
    ps.m = 1

    # Define the log-likelihood of the data given the model
    loglike = PSDLogLikelihood(ps.freq, ps.power, observation_model, m=ps.m)
    # Parameter estimation object
    fm = "L-BFGS-B"
    parameter_estimate = PSDParEst(ps, fitmethod=fm, max_post=False)

    # Estimate the starting parameters
    ipe = InitialParameterEstimatePlC(ps.freq, ps.power, ir=(0, 10), ar=(0, 5), br=(-50, -1))
    return parameter_estimate.fit(loglike, [ipe.amplitude, ipe.index, ipe.background],
                                  scipy_optimize_options=scipy_optimize_options)


if __name__ == '__main__':
    # Get the minimum and maximum spectral power across the study types and waves
    min_spectral_power = 1e20
    max_spectral_power = 0.0
    for study_type in study_types:
        for wave in waves:

            # branch location
            b = [ds.corename, ds.original_datatype, wave]

            # Directory
            directory = ds.datalocationtools.save_location_calculator(ds, b)["project_data"]

            # The spectral power
            spectral_power = (load_spectrum(directory, study_type, wave, window, power_type))['arr_0']
            this_min_power = np.nanmin(spectral_power)
            this_max_power = np.nanmax(spectral_power)
            if this_min_power < min_spectral_power:
                min_spectral_power = this_min_power
            if this_max_power > max_spectral_power:
                max_spectral_power = this_max_power
    print('Minimum spectral power = ' + str(min_spectral_power))
    print('Maximum spectral power = ' + str(max_spectral_power))

    # Now do the fitting
    cluster = LocalCluster()
    for study_type in study_types:
        for wave in waves:
            # General notification that we have a new data-set
            print('\nLoading new data for fitting')

            # branch location
            b = [ds.corename, ds.original_datatype, wave]

            # Directory
            directory = ds.datalocationtools.save_location_calculator(ds, b)["project_data"]

            # Location of the project data
            for_analysis = load_spectrum(ds, b, study_type, wave, window, power_type)

            # Create a list of power spectra for use by the fitter and Dask.
            frequencies = for_analysis['arr_1']
            # Normalize the frequencies?
            if normalize_frequencies:
                frequencies = frequencies / frequencies[0]

            # Analyze a smaller portion of the data for testing purposes?
            shape = (for_analysis['arr_0']).shape
            x0 = subsection[0][0] if subsection[0][0] is not None else 0
            x1 = subsection[0][1] if subsection[0][1] is not None else shape[0]
            y0 = subsection[1][0] if subsection[1][0] is not None else 0
            y1 = subsection[1][1] if subsection[1][1] is not None else shape[1]
            data = (for_analysis['arr_0'])[x0:x1, y0:y1, :]
            mfits = np.zeros_like(data)
            shape = data.shape
            nx = shape[0]
            ny = shape[1]
            powers = list()
            for i in range(0, nx):
                for j in range(0, ny):
                    if divide_by_initial_power:
                        norm = data[i, j, 0]
                    else:
                        norm = 1.0
                    powers.append((frequencies, data[i, j, :]/norm))

            # Use Dask to to fit the spectra
            # client = distributed.Client()
            print('Dask processing of {:n} spectra'.format(nx*ny))
            client = Client(cluster)

            # Get the start time
            t_start = Time.now()
            print(f'...started Dask processing at {t_start}')

            # Do the model fits
            results = client.map(dask_fit_fourier_pl_c, powers)
            z = client.gather(results)

            # Get the end time and calculate the time taken
            t_end = Time.now()
            print(f'...ended Dask processing at {t_end}')
            print('Time taken to perform fits', (t_end-t_start).to(u.s))

            # Now go through all the results and save out the results
            # Total number of outputs = 2*n_parameters + 3
            # For short hand call n_parameters 'n' instead
            # 0 : n-1 : parameter values
            # n : 2n-1 : error in parameter values
            # 2n + 0 : AIC
            # 2n + 1 : BIC
            # 2n + 2 : result
            n_parameters = len(z[0].p_opt)
            n_outputs = 2 * n_parameters + 3
            outputs = np.zeros((nx, ny, n_outputs))

            # Turn the results into an easier to use array
            for i in range(0, nx):
                for j in range(0, ny):
                    index = j + i*ny
                    r = z[index]
                    outputs[i, j, 0:n_parameters] = r.p_opt[:]
                    outputs[i, j, n_parameters:2*n_parameters] = r.err[:]
                    outputs[i, j, 2 * n_parameters + 0] = r.aic
                    outputs[i, j, 2 * n_parameters + 1] = r.bic
                    outputs[i, j, 2 * n_parameters + 2] = r.result

                    mfits[i, j, :] = r.mfit[:]

            filename = '{:s}_{:s}_{:s}_{:s}.{:s}.outputs.step3.npz'.format(observation_model.name, study_type, wave, window, power_type)
            filepath = os.path.join(directory, filename)
            print('Saving ' + filepath)
            np.savez(filepath, outputs)

            filename = '{:s}_{:s}_{:s}_{:s}.{:s}.mfits.step3.npz'.format(observation_model.name, study_type, wave, window, power_type)
            filepath = os.path.join(directory, filename)
            print('Saving ' + filepath)
            np.savez(filepath, mfits, frequencies)

            filename = '{:s}_{:s}_{:s}_{:s}.{:s}.analysis.step3.npz'.format(observation_model.name, study_type, wave, window, power_type)
            filepath = os.path.join(directory, filename)
            print('Saving ' + filepath)
            np.savez(filepath, [x0, x1, y0, y1], normalize_frequencies, divide_by_initial_power)

            # Create a list the names of the output in the same order that they appear in the outputs
            output_names = list()
            param_names = observation_model.param_names
            fixed = observation_model.fixed
            for name in param_names:
                if not fixed[name]:
                    output_names.append(name)
            for name in param_names:
                if not fixed[name]:
                    output_names.append('err_{:s}'.format(name))
            output_names.append('aic')
            output_names.append('bic')
            output_names.append('result')

            filename = '{:s}_{:s}_{:s}_{:s}.{:s}.names.step3.txt'.format(observation_model.name, study_type, wave, window, power_type)
            filepath = os.path.join(directory, filename)
            print('Saving ' + filepath)
            with open(filepath, 'w') as file_out:
                for output_name in output_names:
                    file_out.write(f"{output_name}\n")

            # Restart the client
            client.restart()
