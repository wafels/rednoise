""" 
The purpose of this program is to analyze test data of a given power law index
and duration and find the probability distribution of measured power law
indices.
"""  
 
import numpy as np
import os
from rnfit2 import Do_MCMC
from rnsimulation import TimeSeries, SimplePowerLawSpectrumWithConstantBackground, TimeSeriesFromPowerSpectrum
from matplotlib import pyplot as plt
from pymcmodels import single_power_law_with_constant
#
dt = 12.0
nt = 300
pls = SimplePowerLawSpectrumWithConstantBackground([10.0, 2.0, -5.0], nt=nt, dt=dt)
data = TimeSeriesFromPowerSpectrum(pls).sample
ts = TimeSeries(dt * np.arange(0, nt), data)

z = Do_MCMC([ts]).okgo(single_power_law_with_constant, iter=50000, burn=10000, thin=5, progress_bar=False)
