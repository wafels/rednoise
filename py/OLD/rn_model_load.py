"""
A PyMC model for observed spectra.
"""

import pymc
import numpy as np
import matplotlib.pyplot as plt
import rn_utils 



def rn_model_load(analysis_frequencies, analysis_power):

#    __all__ = ['analysis_power', 'analysis_frequencies', 'power_law_index',
#           'power_law_norm', 'power_law_spectrum', 'spectrum']

    
    estimate = rn_utils.do_simple_fit(analysis_frequencies, analysis_power)
    
    c_estimate = estimate[0]
    m_estimate = estimate[1]
    
    # Define data and stochastics
    @pymc.stochastic
    power_law_index = pymc.Uniform('power_law_index',
                                   value=m_estimate,
                                   lower=0.0,
                                   upper=m_estimate + 2,
                                   doc='power law index')
    @pymc.stochastic
    power_law_norm = pymc.Uniform('power_law_norm',
                                  value=c_estimate,
                                  lower=c_estimate * 0.8,
                                  upper=c_estimate * 1.2,
                                  doc='power law normalization')
    
    
    # Model for the power law spectrum
    @pymc.deterministic(plot=False)
    def power_law_spectrum(p=power_law_index,
                           a=power_law_norm,
                           f=analysis_frequencies):
        """A pure and simple power law model"""
        out = a * (f ** (-p))
        return out
    
    #@pymc.deterministic(plot=False)
    #def power_law_spectrum_with_constant(p=power_law_index, a=power_law_norm,
    #                                     c=constant, f=frequencies):
    #    """Simple power law with a constant"""
    #    out = empty(frequencies)
    #    out = c + a/(f**p)
    #    return out
    
    #@pymc.deterministic(plot=False)
    #def broken_power_law_spectrum(p2=power_law_index_above,
    #                              p1=power_law_index_below,
    #                              bf=break_frequency,
    #                              a=power_law_norm,
    #                              f=analysis_frequencies):
    #    """A broken power law model"""
    #    out = np.empty(len(f))
    #    out[f < bf] = a * (f[f < bf] ** (-p1))
    #    out[f > bf] = a * (f[f >= bf] ** (-p2)) * bf ** (p2 - p1)
    #    return out
    
    # This is the PyMC model we will use: fits the model defined in
    # beta=1.0 / model to the power law spectrum we are analyzing
    # value=analysis_power
    
    spectrum = pymc.Exponential('spectrum',
                           beta=1.0 / power_law_spectrum,
                           value=analysis_power,
                           observed=True)
    return locals()