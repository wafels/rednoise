"""
The purpose of this program is to analyze test data of a given power law index
and duration and find the probability distribution of measured power law
indices.
"""

import numpy as np 
from timeseries import TimeSeries
from rnsimulation import  SimplePowerLawSpectrum, TimeSeriesFromPowerSpectrum
from matplotlib import pyplot as plt
from rnfit2 import Do_MCMC
from pymcmodels import single_power_law_with_constant_not_normalized
#import ppcheck

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
    data3 = data3 + tsnew3.sample# * np.random.uniform()

ts3 = TimeSeries(dt * np.arange(0, nt), data3)


this = ([ts3.pfreq, ts3.ppower],)

norm_estimate = np.zeros((3,))
norm_estimate[0] = ts3.ppower[0]
norm_estimate[1] = norm_estimate[0] / 1000.0
norm_estimate[2] = norm_estimate[0] * 1000.0

background_estimate = np.zeros_like(norm_estimate)
background_estimate[0] = np.mean(ts3.ppower[-10:-1])
background_estimate[1] = background_estimate[0] / 1000.0
background_estimate[2] = background_estimate[0] * 1000.0

estimate = {"norm_estimate": norm_estimate,
            "background_estimate": background_estimate}


"""
plt.figure(1)
plt.loglog(ts0.PowerSpectrum.frequencies.positive, pwr0, label='V=1, W=1')
plt.loglog(ts1.PowerSpectrum.frequencies.positive, pwr1, label='V=%4i, W=1' %(V))
plt.loglog(ts2.PowerSpectrum.frequencies.positive, pwr2, label='V=1, W=%4i' %(W))
plt.loglog(ts3.PowerSpectrum.frequencies.positive, pwr3, label='V=%4i, W=%4i' %(V, W))
plt.legend()
plt.show()
"""
z = Do_MCMC(this).okgo(single_power_law_with_constant_not_normalized, estimate=estimate,
                       iter=50000, burn=10000, thin=5, progress_bar=True)




#z = Do_MCMC([ts]).okgo(single_power_law_with_constant, iter=50000, burn=10000, thin=5, progress_bar=False)
