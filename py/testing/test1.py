"""
Test the power spectrum tools
"""

# Test 1
import rnspectralmodels
import rnsimulation
import rnfit
from matplotlib import pyplot as plt
import numpy as np
plt.ion()

nt = 3000
dt = 12.0
const = 0.0
index = 2.0

parameters = np.asarray([const, index])

f = rnsimulation.equally_spaced_nonzero_frequencies(nt, dt)
p = rnspectralmodels.power_law(f, parameters)

#plt.figure(1)
#plt.loglog(f, p)
#plt.show()

P = rnsimulation.SimplePowerLawSpectrum(parameters, nt=nt, dt=dt)

z = rnsimulation.TimeSeriesFromPowerSpectrum(P, V=100, W=100).sample

pwr = (np.absolute(np.fft.fft(z))) ** 2



coefficients = rnfit.simple_fit(z, dt)
print coefficients

# Define the white noise power spectrum
P = rnsimulation.ConstantSpectrum(parameters, nt=nt, dt=dt)

# Get a time series that is noisy
white_noise = rnsimulation.TimeSeriesFromPowerSpectrum(P, V=100, W=100).sample

# Sample times
t = dt * np.arange(0, nt)

# Calculate a simulated and noisy time series
time_series = oscillation(A_osc, B_osc, frequency, t) + \
              trend(polynomial, t) + \
              white_noise


def oscillation(A, B, frequency, t):
    """
    Define a single sinusoidal oscillation
    """
    return A * np.sin(2 * np.pi * frequency * t) + \
        B * np.cos(2 * np.pi * frequency * t)


def trend(polynomial, t):
    """
    Define a polynomial background trend
    """
    return np.polyval(polynomial, t)