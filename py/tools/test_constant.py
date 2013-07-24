""" 
The purpose of this program is to analyze test data of a given power law index
and duration and find the probability distribution of measured power law
indices.
""" 
 
import numpy as np
import rnfit, pymc_models
from rnsimulation import ConstantSpectrum, TimeSeriesFromPowerSpectrum
from matplotlib import pyplot as plt

#
dt=12.0

pls = ConstantSpectrum( 1.0, nt=300, dt=dt)
data = TimeSeriesFromPowerSpectrum(pls).sample

observed_power_spectrum = (np.absolute(np.fft.fft(data/np.std(data)))) ** 2
fftfreq = np.fft.fftfreq(300, dt)
analysis_frequencies = fftfreq[fftfreq >= 0][1:-1]
analysis_power = observed_power_spectrum[fftfreq >= 0][1:-1]

x = np.log(analysis_frequencies)
y = np.log(analysis_power)
coefficients = np.polyfit(x, y, 1)
m_estimate = -coefficients[0]
c_estimate = np.exp(coefficients[1])



# Do the fit
zzz = rnfit.dopymc(pymc_models.spl(), data, dt)

zzz.M1.sample(iter=50000, burn=10000, thin=10)

plt.loglog(zzz.obs_freq, zzz.obs_pwr)

# The result contains the PyMC sample object and the
ppp = np.mean(M1.trace("power_law_index")[:])
nnn = np.mean(M1.trace("power_law_norm")[:])

