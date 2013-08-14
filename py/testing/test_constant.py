""" 
The purpose of this program is to analyze test data of a given power law index
and duration and find the probability distribution of measured power law
indices.
"""

import rnfit
from rnsimulation import ConstantSpectrum, TimeSeriesFromPowerSpectrum, SimplePowerLawSpectrumWithConstantBackground
from pymcmodels import single_power_law_with_constant

#
dt = 12.0

pls = ConstantSpectrum([1.0], nt=300, dt=dt)
data = TimeSeriesFromPowerSpectrum(pls).sample
for i in range(0, 200):
    data = data + TimeSeriesFromPowerSpectrum(pls).sample

z = rnfit.Do_MCMC([data], dt=dt).okgo(single_power_law_with_constant, iter=50000, burn=10000, thin=5, progress_bar=False)
