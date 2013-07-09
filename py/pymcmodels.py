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


def single_power_law(analysis_frequencies, analysis_power, m_estimate):
    """Set up a PyMC model: power law for the power spectrun"""
    
    # PyMC definitions
    # Define data and stochastics
    power_law_index = pymc.Uniform('power_law_index',
                                   lower=-1.0,
                                   upper=m_estimate + 2,
                                   doc='power law index')

    power_law_norm = pymc.Uniform('power_law_norm',
                                  lower=-100.0,
                                  upper=100.0,
                                  doc='power law normalization')

    # Model for the power law spectrum
    @pymc.deterministic(plot=False)
    def fourier_power_spectrum(p=power_law_index,
                           a=power_law_norm,
                           f=analysis_frequencies):
        """A pure and simple power law model"""
        out = rnspectralmodels.power_law(f, [a, p])
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
                                   upper=6.0,
                                   doc='power law index')

    power_law_norm = pymc.Uniform('power_law_norm',
                                  lower=-100.0,
                                  upper=100.0,
                                  doc='power law normalization')

    background = pymc.Uniform('background',
                                  lower=-100.0,
                                  upper=100.0,
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

    # MCMC model as a list
    return [power_law_index, power_law_norm, background, fourier_power_spectrum, spectrum]


def broken_power_law(analysis_frequencies, analysis_power,
                                a_est, d1_est, d2_est, f_break):
    """Set up a PyMC model: broken power law for the power spectrun"""
        
    # PyMC definitions
    # Define data and stochastics
    power_law_norm = pymc.Uniform('power_law_norm',
                                  lower=-100,
                                  upper=100.0,
                                  doc='power law normalization')
    
    delta1 = pymc.Uniform('delta1',
                          lower=0.0,
                          upper=6.0,
                          doc='delta1')

    delta2 = pymc.Uniform('delta2',
                          lower=-1.0,
                          upper=6.0,
                          doc='delta2')

    breakf = pymc.Uniform('breakf',
                          lower=np.log10(analysis_frequencies[0]),
                          upper=np.log10(analysis_frequencies[-1]),
                          doc='break frequency')

    # Model for the power law spectrum
    @pymc.deterministic(plot=False)
    def fourier_power_spectrum(d1=delta1, d2=delta2, lnf0=breakf,
                           a=power_law_norm,
                           f=analysis_frequencies):
        """A pure and simple broken power law model"""
        return rnspectralmodels.broken_power_law(f, [a, d1, d2, lnf0])

    spectrum = pymc.Exponential('spectrum',
                           beta=1.0 / fourier_power_spectrum,
                           value=analysis_power,
                           observed=True)

    return [delta1, delta2, breakf, power_law_norm, fourier_power_spectrum, spectrum]


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


    