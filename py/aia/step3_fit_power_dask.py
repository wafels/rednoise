#
# Step 3.  Load in the FFT power spectra and fit models. Save the results
#
import os
import numpy as np

import distributed

from stingray import Powerspectrum
from stingray.modeling import PSDLogLikelihood
from stingray.modeling import PSDParEst

import astropy.units as u
from astropy.time import Time
from spectral_model_parameter_estimators import InitialParameterEstimatePlC
from details_spectral_models import SelectModel
import details_study as ds

# Data to analyze
# Wavelengths we want to analyze
waves = ['94', '171', '131', '193', '211', '335']
waves = ['171']

# Type of power spectrum
power_type = 'absolute'

# Window used
window = 'hanning'

# Define the observation model
this_model = SelectModel('pl_c')
observation_model = this_model.observation_model
scipy_optimize_options = this_model.scipy_optimize_options


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
    parameter_estimate = PSDParEst(ps, fitmethod="L-BFGS-B", max_post=False)

    # Estimate the starting parameters
    ipe = InitialParameterEstimatePlC(ps.freq, ps.power)
    return parameter_estimate.fit(loglike, [ipe.amplitude, ipe.index, ipe.background],
                                  scipy_optimize_options=scipy_optimize_options)


if __name__ == '__main__':

    for wave in waves:
        # General notification that we have a new data-set
        print('\nLoading New Data')

        # branch location
        b = [ds.corename, ds.original_datatype, wave]

        # Region identifier name
        region_id = ds.datalocationtools.ident_creator(b)

        # Location of the project data
        directory = ds.datalocationtools.save_location_calculator(ds.roots, b)["project_data"]
        filename = '{:s}_{:s}_{:s}.{:s}.step2.npz'.format(ds.study_type, wave, window, power_type)
        filepath = os.path.join(directory, filename)
        print('Loading power spectra from ' + filepath)
        for_analysis = np.load(filepath)

        # Create a list of power spectra for use by the fitter and Dask.
        frequencies = for_analysis['arr_1']
        data = for_analysis['arr_0']
        mfits = np.zeros_like(data)
        shape = data.shape
        nx = shape[0]
        ny = shape[1]
        powers = list()
        for i in range(0, nx):
            for j in range(0, ny):
                powers.append((frequencies, data[i, j, :]))

        # Use Dask to to fit the spectra
        client = distributed.Client()
        print('Dask processing of {:n} spectra'.format(nx*ny))

        # Get the start time
        t_start = Time.now()

        # Do the model fits
        results = client.map(dask_fit_fourier_pl_c, powers)
        z = client.gather(results)

        # Get the end time and calculate the time taken
        t_end = Time.now()
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

        filename = '{:s}_{:s}_{:s}_{:s}.{:s}.outputs.step3.npz'.format(observation_model.name, ds.study_type, wave, window, power_type)
        filepath = os.path.join(directory, filename)
        print('Saving ' + filepath)
        np.savez(filepath, outputs)

        filename = '{:s}_{:s}_{:s}_{:s}.{:s}.mfits.step3.npz'.format(observation_model.name, ds.study_type, wave, window, power_type)
        filepath = os.path.join(directory, filename)
        print('Saving ' + filepath)
        np.savez(filepath, mfits, frequencies)

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

        filename = '{:s}_{:s}_{:s}_{:s}.{:s}.names.step3.txt'.format(observation_model.name, ds.study_type, wave, window, power_type)
        filepath = os.path.join(directory, filename)
        print('Saving ' + filepath)
        with open(filepath, 'w') as file_out:
            for output_name in output_names:
                file_out.write(f"{output_name}\n")
