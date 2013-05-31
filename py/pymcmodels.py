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

def single_power_law(analysis_frequencies, analysis_power,
                                c_estimate, m_estimate):
    """Set up a PyMC model: power law for the power spectrun"""
    
    # PyMC definitions
    # Define data and stochastics
    power_law_index = pymc.Uniform('power_law_index',
                                   lower=-1.0,
                                   upper=m_estimate + 2,
                                   doc='power law index')

    power_law_norm = pymc.Uniform('power_law_norm',
                                  lower=c_estimate * 0.01,
                                  upper=c_estimate * 100.0,
                                  doc='power law normalization')

    # Model for the power law spectrum
    @pymc.deterministic(plot=False)
    def fourier_power_spectrum(p=power_law_index,
                           a=power_law_norm,
                           f=analysis_frequencies):
        """A pure and simple power law model"""
        out = a * (f ** (-p))
        return out

    spectrum = pymc.Exponential('spectrum',
                           beta=1.0 / fourier_power_spectrum,
                           value=analysis_power,
                           observed=True)

    # MCMC model as a list
    return [power_law_index, power_law_norm, fourier_power_spectrum, spectrum]



def single_power_law_with_constant(analysis_frequencies, analysis_power,
                                c_estimate, m_estimate, b_estimate):
    """Set up a PyMC model: power law for the power spectrun"""
    
    # PyMC definitions
    # Define data and stochastics
    power_law_index = pymc.Uniform('power_law_index',
                                   lower=-1.0,
                                   upper=m_estimate + 2,
                                   doc='power law index')

    power_law_norm = pymc.Uniform('power_law_norm',
                                  lower=c_estimate * 0.0001,
                                  upper=c_estimate * 10000.0,
                                  doc='power law normalization')

    background = pymc.Uniform('background',
                                  lower=b_estimate * 0.0001,
                                  upper=b_estimate * 10000.0,
                                  doc='background')

    # Model for the power law spectrum
    @pymc.deterministic(plot=False)
    def fourier_power_spectrum(p=power_law_index,
                               a=power_law_norm,
                               b=background,
                               f=analysis_frequencies):
        """A pure and simple power law model"""
        out = a * (f ** (-p)) + b
        return out

    spectrum = pymc.Exponential('spectrum',
                           beta=1.0 / fourier_power_spectrum,
                           value=analysis_power,
                           observed=True)

    # MCMC model as a list
    return [power_law_index, power_law_norm, background, fourier_power_spectrum, spectrum]


def broken_power_law(analysis_frequencies, analysis_power,
                                a_est, d1_est, d2_est, f_break):
    """Set up a PyMC model: broken power law for the power spectrun"""
        
    # PyMC definitions
    # Define data and stochastics
    power_law_norm = pymc.Uniform('power_law_norm',
                                  lower=a_est * 0.01,
                                  upper=a_est * 100.0,
                                  doc='power law normalization')
    
    delta1 = pymc.Uniform('delta1',
                          lower=-1.0,
                          upper=d1_est + 2,
                          doc='delta1')

    delta2 = pymc.Uniform('delta2',
                          lower=-1.0,
                          upper=d2_est + 2,
                          doc='delta2')

    breakf = pymc.Uniform('delta2',
                          lower=analysis_frequencies[0],
                          upper=analysis_frequencies[-1],
                          doc='break frequency')

    # Model for the power law spectrum
    @pymc.deterministic(plot=False)
    def fourier_power_spectrum(d1=delta1, d2=delta2, f0=breakf,
                           a=power_law_norm,
                           f=analysis_frequencies):
        """A pure and simple power law model"""
        out = np.zeros(shape=f.shape)
        out[f<f0] = a * (f ** (-d1))
        out[f>=f0] = a * (f ** (-d2)) * (f0 ** (d2 - d1))
        return out

    spectrum = pymc.Exponential('spectrum',
                           beta=1.0 / fourier_power_spectrum,
                           value=analysis_power,
                           observed=True)
    return


def broken_power_law_delta_oscillation():
    """Set up a PyMC model: broken power law for the power spectrun with
    a single monochromatic sinusoidal oscillation present"""
    pass
    return


def broken_power_law_boxcar_oscillation():
    """Set up a PyMC model: broken power law for the power spectrun with
    oscillations present.  In the power spectrum, these oscillations appear
    as a boxcar of power"""
    pass
    return


def broken_power_law_gaussian_oscillation():
    """Set up a PyMC model: broken power law for the power spectrun with
    oscillations present.  In the power spectrum, these oscillations appear
    with a Gaussian-shaped profile of power"""
    pass
    return


    