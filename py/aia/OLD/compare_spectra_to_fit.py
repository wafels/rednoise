#
# Compare the fit spectrum to the original spectrum
#
import os
import numpy as np

import matplotlib.pyplot as plt

from astropy.modeling.fitting import _fitter_to_model_params
from astropy.modeling import models

from stingray import Powerspectrum
from stingray.modeling import PSDLogLikelihood
from stingray.modeling import PSDParEst

from spectral_model_parameter_estimators import InitialParameterEstimatePlC
import details_study as ds

# Data to analyze
# Wavelengths we want to analyze
waves = ['171']

# Type of power spectrum
power_type = 'absolute'

# Window used
window = 'hanning'

# Power spectrum mode;
# Power law component
power_law = models.PowerLaw1D()

# fix x_0 of power law component
power_law.x_0.fixed = True

# Constant component
constant = models.Const1D()

# Define the observation model
observation_model = power_law + constant
observation_model.name = 'pl_c'


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
    return parameter_estimate.fit(loglike, [ipe.amplitude, ipe.index, ipe.background])


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
        print('Loading observed power spectra from ' + filepath)
        for_analysis = np.load(filepath)

        # Get the frequencies and the data we are interested in
        frequencies = for_analysis['arr_1']
        x_min = 100  # 0
        x_max = 110  # nx
        y_min = 100  # 0
        y_max = 110  # ny
        data = for_analysis['arr_0'][x_min:x_max, y_min:y_max]
        shape = data.shape
        nx = shape[0]
        ny = shape[1]

        # Load the model spectra parameters
        filename = '{:s}_{:s}_{:s}_{:s}.{:s}.step3.npz'.format(observation_model.name, ds.study_type, wave, window, power_type)
        filepath = os.path.join(directory, filename)
        print('Loading model parameters ' + filepath)
        outputs = np.load(filepath)['arr_0']

        # Load the parameter names
        filename = '{:s}_{:s}_{:s}_{:s}.{:s}.names.step3.npz'.format(observation_model.name, ds.study_type, wave, window, power_type)
        filepath = os.path.join(directory, filename)
        with open(filepath) as f:
            output_names = [line.rstrip() for line in f]

        # Create the fit spectrum
        xx = 5
        yy = 5
        _fitter_to_model_params(observation_model,
                                [outputs[xx, yy, 0], outputs[xx, yy, 1], outputs[xx, yy, 2]])
        # Create the true data
        psd_shape = observation_model(frequencies)

        plt.loglog(frequencies, data[xx, yy, :])
        plt.loglog(frequencies, psd_shape)
