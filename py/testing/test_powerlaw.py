"""
The purpose of this program is to analyze test data of a given power law index
and duration and find the probability distribution of measured power law
indices.
"""

import numpy as np
from rnsimulation import TimeSeries, SimplePowerLawSpectrum, TimeSeriesFromPowerSpectrum
from matplotlib import pyplot as plt
from rnfit2 import Do_MCMC
from pymcmodels import single_power_law

#
dt = 12.0
nt = 300
V = 2
W = 3
no_noise = False
pls = SimplePowerLawSpectrum([10.0, 0.1], nt=nt, dt=dt)
tsnew0 = TimeSeriesFromPowerSpectrum(pls, V=1, W=1, no_noise=no_noise)
tsnew1 = TimeSeriesFromPowerSpectrum(pls, V=V, W=1, no_noise=no_noise)
tsnew2 = TimeSeriesFromPowerSpectrum(pls, V=1, W=W, no_noise=no_noise)
tsnew3 = TimeSeriesFromPowerSpectrum(pls, V=V, W=W, no_noise=no_noise)

data0 = tsnew0.sample
ts0 = TimeSeries(dt * np.arange(0, nt), data0)
pwr0 = (np.abs(np.fft.fft(data0)) ** 2)[ts0.PowerSpectrum.frequencies.posindex]
pwr0 = pwr0 / np.mean(pwr0)
pwr0 = pwr0 / np.std(pwr0)

data1 = tsnew1.sample
ts1 = TimeSeries(dt * np.arange(0, nt), data1)
pwr1 = (np.abs(np.fft.fft(data1)) ** 2)[ts1.PowerSpectrum.frequencies.posindex]
pwr1 = pwr1 / np.mean(pwr1)
pwr1 = pwr1 / np.std(pwr1)

data2 = tsnew2.sample
ts2 = TimeSeries(dt * np.arange(0, nt), data2)
pwr2 = (np.abs(np.fft.fft(data2)) ** 2)[ts2.PowerSpectrum.frequencies.posindex]
pwr2 = pwr2 / np.mean(pwr2)
pwr2 = pwr2 / np.std(pwr2)

data3 = tsnew3.sample
ts3 = TimeSeries(dt * np.arange(0, nt), data3)
pwr3 = (np.abs(np.fft.fft(data3)) ** 2)[ts3.PowerSpectrum.frequencies.posindex]
pwr3 = pwr3 / np.mean(pwr3)
pwr3 = pwr3 / np.std(pwr3)
"""
plt.figure(1)
plt.loglog(ts0.PowerSpectrum.frequencies.positive, pwr0, label='V=1, W=1')
plt.loglog(ts1.PowerSpectrum.frequencies.positive, pwr1, label='V=%4i, W=1' %(V))
plt.loglog(ts2.PowerSpectrum.frequencies.positive, pwr2, label='V=1, W=%4i' %(W))
plt.loglog(ts3.PowerSpectrum.frequencies.positive, pwr3, label='V=%4i, W=%4i' %(V, W))
plt.legend()
plt.show()
"""
z = Do_MCMC([ts0, ts1, ts2, ts3]).okgo(single_power_law, iter=50000, burn=10000, thin=5, progress_bar=False)

#z = Do_MCMC([ts]).okgo(single_power_law_with_constant, iter=50000, burn=10000, thin=5, progress_bar=False)
