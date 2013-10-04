"""
Test the power spectrum tools
"""

# Test 1
import rnspectralmodels
import rnsimulation
from matplotlib import pyplot as plt
import numpy as np

plt.ion()

nt = 300
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

z = rnsimulation.TimeSeriesFromPowerSpectrum(P, V=10, W=10).sample

#pwr = (np.absolute(np.fft.fft(z))) ** 2
