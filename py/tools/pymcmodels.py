"""
PyMC models for the red noise study.

All the models have two objects in common:

(1) a function called "fourier_power_spectrum" which is the theoretical
    description of the observed spectrum,
(2) an object "spectrum" which is likelihood of the observed spectrum given
    the model.
"""

"""
Notes

By observation, it seems that the power laws for the AIA data are distributed
approximately as follows

iobs - (average power spectra, averaged over all the pixels)
 - lognormally dsitributed

logiobs
 - normally distributed.

Hence we set up PyMC models that implement these distributions.




"""


import numpy as np
import pymc
import rnspectralmodels
import rnsimulation


#
# Fit a power law with a constant using PyMC.  The original application for
# this was to fit power laws to Fourier power spectra which are exponentially
# distributed
#
def single_power_law_with_constant(analysis_frequencies,
                                   analysis_power,
                                   likelihood_type='Exponential',
                                   **kwargs):
    """Set up a PyMC model: power law for the power spectrum"""

    # PyMC definitions
    # Define data and stochastics
    power_law_index = pymc.Uniform('power_law_index',
                                   lower=-1.0,
                                   upper=6.0,
                                   doc='power law index')

    power_law_norm = pymc.Uniform('power_law_norm',
                                  lower=-20.0,
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

    #
    # Exponential distribution
    #
    if likelihood_type == 'Exponential':
        spectrum = pymc.Exponential('spectrum',
                               beta=1.0 / fourier_power_spectrum,
                               value=analysis_power,
                               observed=True)

        predictive = pymc.Exponential('predictive', beta=1.0 / fourier_power_spectrum)

    #
    # Assumes that the input data is normally distributed
    #
    if likelihood_type == 'Normal':
        if "sigma" not in kwargs:
            raise ValueError
        else:
            print('Likelihood=' + likelihood_type)
            spectrum = pymc.Normal('spectrum',
                                   tau=1.0 / (kwargs['sigma'] ** 2),
                                   mu=fourier_power_spectrum,
                                   value=analysis_power,
                                   observed=True)
            predictive = pymc.Normal('predictive',
                                     tau=1.0 / (kwargs['sigma'] ** 2),
                                     mu=fourier_power_spectrum)

    #
    # Assumes that the input data is lognormally distributed
    #
    if likelihood_type == 'Lognormal':
        if "sigma" not in kwargs:
            raise ValueError
        else:
            print('Likelihood=' + likelihood_type)
            spectrum = pymc.Lognormal('spectrum',
                                   tau=1.0 / (kwargs['sigma'] ** 2),
                                   mu=fourier_power_spectrum,
                                   value=analysis_power,
                                   observed=True)
            predictive = pymc.lognormal('predictive',
                                     tau=1.0 / (kwargs['sigma'] ** 2),
                                     mu=fourier_power_spectrum)

    # MCMC model
    return locals()


def splwc_GaussianBump(analysis_frequencies, analysis_power):
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
                                  lower=-20.0,
                                  upper=10.0,
                                  doc='background')

    gaussian_amplitude = pymc.Uniform('gaussian_amplitude',
                                  lower=0.0,
                                  upper=10.0,
                                  doc='gaussian_amplitude')

    gaussian_position = pymc.Uniform('gaussian_position',
                                  lower=0.01,
                                  upper=10.0,
                                  doc='gaussian_position')

    gaussian_width = pymc.Uniform('gaussian_width',
                                  lower=0.001,
                                  upper=1.0 * np.log(10.0),
                                  doc='gaussian_width')

    # Model for the power law spectrum
    @pymc.deterministic(plot=False)
    def fourier_power_spectrum(p=power_law_index,
                               a=power_law_norm,
                               b=background,
                               ga=gaussian_amplitude,
                               gc=gaussian_position,
                               gs=gaussian_width,
                               f=analysis_frequencies):
        """A pure and simple power law model"""
        out = rnspectralmodels.splwc_GaussianBump(f, [a, p, b, ga, gc, gs])
        return out

    spectrum = pymc.Exponential('spectrum',
                           beta=1.0 / fourier_power_spectrum,
                           value=analysis_power,
                           observed=True)

    predictive = pymc.Exponential('predictive', beta=1.0 / fourier_power_spectrum)

    # MCMC model
    return locals()


def Log_splwc_GaussianBump(analysis_frequencies, analysis_power, sigma):
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
                                  lower=-20.0,
                                  upper=10.0,
                                  doc='background')

    gaussian_amplitude = pymc.Uniform('gaussian_amplitude',
                                  lower=0.0,
                                  upper=10.0,
                                  doc='gaussian_amplitude')

    gaussian_position = pymc.Uniform('gaussian_position',
                                  lower=0.01,
                                  upper=6.0,
                                  doc='gaussian_position')

    gaussian_width = pymc.Uniform('gaussian_width',
                                  lower=0.001,
                                  upper=1.0 * np.log(10.0),
                                  doc='gaussian_width')

    # Model for the power law spectrum
    @pymc.deterministic(plot=False)
    def fourier_power_spectrum(p=power_law_index,
                               a=power_law_norm,
                               b=background,
                               ga=gaussian_amplitude,
                               gc=gaussian_position,
                               gs=gaussian_width,
                               f=analysis_frequencies):
        """A pure and simple power law model"""
        out = rnspectralmodels.Log_splwc_GaussianBump(f, [a, p, b, ga, gc, gs])
        return out

    spectrum = pymc.Normal('spectrum',
                           tau=1.0 / (sigma ** 2),
                           mu=fourier_power_spectrum,
                           value=analysis_power,
                           observed=True)

    predictive = pymc.Normal('predictive',
                             tau = 1.0 / (sigma ** 2),
                             mu=fourier_power_spectrum)

    # MCMC model
    return locals()


def Log_splwc(analysis_frequencies, analysis_power, sigma):
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
                                  lower=-20.0,
                                  upper=10.0,
                                  doc='background')

    # Model for the power law spectrum
    @pymc.deterministic(plot=False)
    def fourier_power_spectrum(p=power_law_index,
                               a=power_law_norm,
                               b=background,
                               f=analysis_frequencies):
        """A pure and simple power law model"""
        out = rnspectralmodels.Log_splwc(f, [a, p, b])
        return out

    spectrum = pymc.Normal('spectrum',
                           tau = 1.0 / (sigma ** 2),
                           mu=fourier_power_spectrum,
                           value=analysis_power,
                           observed=True)

    predictive = pymc.Normal('predictive',
                             tau = 1.0 / (sigma ** 2),
                             mu=fourier_power_spectrum)

    # MCMC model
    return locals()


#
#
#
def Log_splwc_lognormal(analysis_frequencies, analysis_power, sigma):
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
                                  lower=-20.0,
                                  upper=10.0,
                                  doc='background')

    # Model for the power law spectrum
    @pymc.deterministic(plot=False)
    def fourier_power_spectrum(p=power_law_index,
                               a=power_law_norm,
                               b=background,
                               f=analysis_frequencies):
        """A pure and simple power law model"""
        out = rnspectralmodels.Log_splwc(f, [a, p, b])
        return out

    spectrum = pymc.Lognormal('spectrum',
                           tau=1.0 / (sigma ** 2),
                           mu=fourier_power_spectrum,
                           value=analysis_power,
                           observed=True)

    predictive = pymc.Lognormal('predictive',
                             tau=1.0 / (sigma ** 2),
                             mu=fourier_power_spectrum)

    # MCMC model
    return locals()


def Log_splwc_GaussianBump_lognormal(analysis_frequencies, analysis_power, sigma):
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
                                  lower=-20.0,
                                  upper=10.0,
                                  doc='background')

    gaussian_amplitude = pymc.Uniform('gaussian_amplitude',
                                  lower=0.0,
                                  upper=10.0,
                                  doc='gaussian_amplitude')

    gaussian_position = pymc.Uniform('gaussian_position',
                                  lower=0.01,
                                  upper=6.0,
                                  doc='gaussian_position')

    gaussian_width = pymc.Uniform('gaussian_width',
                                  lower=0.001,
                                  upper=1.0 * np.log(10.0),
                                  doc='gaussian_width')

    # Model for the power law spectrum
    @pymc.deterministic(plot=False)
    def fourier_power_spectrum(p=power_law_index,
                               a=power_law_norm,
                               b=background,
                               ga=gaussian_amplitude,
                               gc=gaussian_position,
                               gs=gaussian_width,
                               f=analysis_frequencies):
        """A pure and simple power law model"""
        out = rnspectralmodels.Log_splwc_GaussianBump(f, [a, p, b, ga, gc, gs])
        return out

    spectrum = pymc.Lognormal('spectrum',
                           tau=1.0 / (sigma ** 2),
                           mu=fourier_power_spectrum,
                           value=analysis_power,
                           observed=True)

    predictive = pymc.Lognormal('predictive',
                             tau=1.0 / (sigma ** 2),
                             mu=fourier_power_spectrum)

    # MCMC model
    return locals()

