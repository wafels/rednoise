"""
The purpose of this program is to analyze test data of a given power law index
and duration and find the probability distribution of measured power law
indices.
"""

import numpy as np
from rnsimulation import TimeSeries, SimplePowerLawSpectrum, TimeSeriesFromPowerSpectrum
from matplotlib import pyplot as plt
#
dt = 12.0
nt = 300
V = 69
W = 1
pls = SimplePowerLawSpectrum([10.0, 2.0], nt=nt, dt=dt)
tsnew1 = TimeSeriesFromPowerSpectrum(pls, V=V, W=1)
tsnew2 = TimeSeriesFromPowerSpectrum(pls, V=1, W=W)

data1 = tsnew1.sample
ts1 = TimeSeries(dt * np.arange(0, nt), data1)

data2 = tsnew2.sample
ts2 = TimeSeries(dt * np.arange(0, nt), data2)

plt.loglog(ts1.PowerSpectrum.frequencies.positive, (V**4)*(np.abs(np.fft.fft(data1)) ** 2)[1:150], label='V='+str(V))
plt.loglog(ts2.PowerSpectrum.frequencies.positive, W*W*(np.abs(np.fft.fft(data2)) ** 2)[1:150], label='W='+str(W))
plt.legend()
plt.show()


#z = Do_MCMC([ts]).okgo(single_power_law_with_constant, iter=50000, burn=10000, thin=5, progress_bar=False)
