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

class spl:
    """
    Single power law model.
    """
    def __init__(self, observed_frequencies=1.0, observed_power=1.0):
        self.observed_frequencies = observed_frequencies
        self.observed_power = observed_power

        # PyMC definitions
        # Define data and stochastics        
        self.power_law_index = pymc.Uniform('power_law_index',
                                       lower=0.0,
                                       upper=6.0,
                                       doc='power law index')
    
        self.power_law_norm = pymc.Uniform('power_law_norm',
                                      lower=-100.0,
                                      upper=100.0,
                                      doc='power law normalization')
    
        # Model for the power law spectrum
        @pymc.deterministic(plot=False)
        def fourier_power_spectrum(p=self.power_law_index,
                                   a=self.power_law_norm,
                                   f=self.observed_frequencies):
            """A pure and simple power law model"""
            out = rnspectralmodels.power_law(f, [a, p])
            return out
    
        self.spectrum = pymc.Exponential('spectrum',
                               beta=1.0 / fourier_power_spectrum,
                               value=observed_power,
                               observed=True)
    
        # MCMC model as a list
        self.pymc_model = [self.power_law_index,
                           self.power_law_norm,
                           fourier_power_spectrum,
                           self.spectrum]