"""
The purpose of this program is to analyze test data of a given power law index
and duration and find the probability distribution of measured power law
indices.
"""

import numpy as np
from rnsimulation import TimeSeries, SimplePowerLawSpectrumWithConstantBackground, TimeSeriesFromPowerSpectrum
from matplotlib import pyplot as plt
import sys

# Create some fake data
dt = 12.0
nt = 300
pls1 = SimplePowerLawSpectrumWithConstantBackground([10.0, 3.0, -5.0], nt=nt, dt=dt)
data = TimeSeriesFromPowerSpectrum(pls1, V=1, W=1).sample
ts = TimeSeries(dt * np.arange(0, nt), data)
