"""
PyMC models for the red noise study.

All the models have two objects in common:

(1) a function called "fourier_power_spectrum" which is the theoretical
    description of the observed spectrum,
(2) an object "spectrum" which is likelihood of the observed spectrum given
    the model.
"""

import numpy as np
import pymc
import rnspectralmodels


def single_power_law_with_constant(analysis_frequencies, analysis_power):
    """Set up a PyMC model: power law for the power spectrum"""

    # PyMC definitions
    # Define data and stochastics
    power_law_index = pymc.Uniform('power_law_index',
                                   lower=-1.0,
                                   upper=6.0,
                                   doc='power law index')

    power_law_norm = pymc.Uniform('power_law_norm',
                                  lower=-10.0,
                                  upper=10.0,
                                  doc='power law normalization')

    background = pymc.Uniform('background',
                                  #value=np.mean(np.log(analysis_power[-10:-1])),
                                  lower=-10.0,
                                  upper=10.0,
                                  doc='background')

    # Model for the power law spectrum
    @pymc.deterministic(plot=False)
    def fourier_power_spectrum(p=power_law_index,
                               a=power_law_norm,
                               b=background,
                               f=analysis_frequencies):
        """A pure and simple power law model"""
        out = rnspectralmodels.power_law_with_constant(f, [a, p, b])
        return out

    spectrum = pymc.Exponential('spectrum',
                           beta=1.0 / fourier_power_spectrum,
                           value=analysis_power,
                           observed=True)

    predictive = pymc.Exponential('predictive', beta=1.0 / fourier_power_spectrum)

    # MCMC model
    return locals()


def single_power_law_with_constant_not_normalized(analysis_frequencies,
                                                  analysis_power,
                                                  estimate):
    """Set up a PyMC model: power law for the power spectrum"""
    norm_estimate = estimate["norm_estimate"]
    background_estimate = estimate["background_estimate"]

    # PyMC definitions
    # Define data and stochastics
    power_law_index = pymc.Uniform('power_law_index',
                                   lower=-1.0,
                                   upper=6.0,
                                   doc='power law index')

    power_law_norm = pymc.Uniform('power_law_norm',
                                  lower=np.log(norm_estimate[1]),
                                  upper=np.log(norm_estimate[2]),
                                  doc='power law normalization')

    background = pymc.Uniform('background',
                                  #value=np.mean(np.log(analysis_power[-10:-1])),
                                  lower=np.log(background_estimate[1]),
                                  upper=np.log(background_estimate[2]),
                                  doc='background')

    # Model for the power law spectrum
    @pymc.deterministic(plot=False)
    def fourier_power_spectrum(p=power_law_index,
                               a=power_law_norm,
                               b=background,
                               f=analysis_frequencies):
        """A pure and simple power law model"""
        out = rnspectralmodels.power_law_with_constant(f, [a, p, b])
        return out

    spectrum = pymc.Exponential('spectrum',
                           beta=1.0 / fourier_power_spectrum,
                           value=analysis_power,
                           observed=True)

    predictive = pymc.Exponential('predictive', beta=1.0 / fourier_power_spectrum)

    # MCMC model
    return locals()

