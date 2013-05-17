"""
A PyMC model for observed spectra.
"""

import pymc
import numpy as np
import matplotlib.pyplot as plt
import rn_utils

__all__ = ['analysis_power', 'analysis_frequencies', 'power_law_index',
           'power_law_norm', 'power_law_spectrum', 'spectrum']

# data goes here
#input_power_spectrum = read in from file???


# Set up the simulated data
n = 300
dt = 12.0
alpha = 2.0

test_data = rn_utils.simulated_power_law(n, dt, alpha)

plt.plot(test_data)
plt.show()

# Power spectrum
observed_power_spectrum = (np.absolute(np.fft.fft(test_data))) ** 2

# Fourier frequencies
fftfreq = np.fft.fftfreq(n, dt)

"""
(1) Conditions on the frequencies...

Drop the zero frequency and the nyquist frequency since these do
not have the exponential distribution.

(2) Conditions on the observed powers

Need to drop the observed power at the zero and nyquist frequencies, since
those frequencies have been dropped.

Would also be good to normalize the power spectrum in some way so that priors
can be more independent of the data.  Note that this can also be handled by
adjusting the prior, rather than normalizing the data.
"""

analysis_frequencies = fftfreq[fftfreq >= 0][1:-1]
analysis_power = observed_power_spectrum[fftfreq >= 0][1:-1]

# get a quick estimate assuming the data is a power law only.
# the [0] entry from lstsq is the gradient, the [1] entry is the
# value at np.log(observed_power_spectrum) = 0.0
coefficients = np.polyfit(np.log(analysis_frequencies),
                           np.log(analysis_power),
                           1)
m_estimate = -coefficients[0]
c_estimate = np.exp(coefficients[1])

# Define data and stochastics
power_law_index = pymc.Uniform('power_law_index',
                               value=m_estimate,
                               lower=0.0,
                               upper=m_estimate + 2,
                               doc='power law index')
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
