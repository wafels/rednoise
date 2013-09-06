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
import ppcheck

#
dt = 12.0
nt = 300
V = 2
W = 3
no_noise = False
pls = SimplePowerLawSpectrum([10.0, 2.0], nt=nt, dt=dt)

tsnew3 = TimeSeriesFromPowerSpectrum(pls, V=V, W=W)

i = 0
ncount = 10000
data3 = np.zeros(shape=(nt))
while i < ncount:
    i = i + 1
    data3 = data3 + tsnew3.sample * np.random.uniform()

ts3 = TimeSeries(dt * np.arange(0, nt), data3)



"""
plt.figure(1)
plt.loglog(ts0.PowerSpectrum.frequencies.positive, pwr0, label='V=1, W=1')
plt.loglog(ts1.PowerSpectrum.frequencies.positive, pwr1, label='V=%4i, W=1' %(V))
plt.loglog(ts2.PowerSpectrum.frequencies.positive, pwr2, label='V=1, W=%4i' %(W))
plt.loglog(ts3.PowerSpectrum.frequencies.positive, pwr3, label='V=%4i, W=%4i' %(V, W))
plt.legend()
plt.show()
"""
z = Do_MCMC([ts3]).okgo(single_power_law, iter=50000, burn=10000, thin=5, progress_bar=False)




#z = Do_MCMC([ts]).okgo(single_power_law_with_constant, iter=50000, burn=10000, thin=5, progress_bar=False)
