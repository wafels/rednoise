"""
The purpose of this program is to analyze test data of a given power law index
and duration and find the probability distribution of measured power law
indices.
"""
 
import numpy as np
from rnsimulation import SimplePowerLawSpectrum, TimeSeriesFromPowerSpectrum
from timeseries import TimeSeries
from matplotlib import pyplot as plt
from rnfit2 import Do_MCMC
from pymcmodels import single_power_law_with_constant_not_normalized
#
dt = 12.0
nt = 300
V = 2
W = 3
no_noise = True
pls = SimplePowerLawSpectrum([10.0, 2.0], nt=nt, dt=dt)
tsnew0 = TimeSeriesFromPowerSpectrum(pls, V=1, W=1, no_noise=no_noise)
tsnew1 = TimeSeriesFromPowerSpectrum(pls, V=V, W=1, no_noise=no_noise)
tsnew2 = TimeSeriesFromPowerSpectrum(pls, V=1, W=W, no_noise=no_noise)
tsnew3 = TimeSeriesFromPowerSpectrum(pls, V=V, W=W, no_noise=no_noise)

rn = TimeSeriesFromPowerSpectrum(pls, V=V, W=W, no_noise=False)
# Add noise
data3 = 100*tsnew3.sample + rn.sample

# Time series object
ts = TimeSeries(dt * np.arange(0, nt), data3)

# Scaled frequency
freqs = ts.PowerSpectrum.frequencies.positive / ts.PowerSpectrum.frequencies.positive[0]

# Form the input for the MCMC algorithm.
this = ([ts.pfreq, ts.ppower],)


norm_estimate = np.zeros((3,))
norm_estimate[0] = ts.ppower[0]
norm_estimate[1] = norm_estimate[0] / 1000.0
norm_estimate[2] = norm_estimate[0] * 1000.0

background_estimate = np.zeros_like(norm_estimate)
background_estimate[0] = np.mean(ts.ppower[-10:-1])
background_estimate[1] = background_estimate[0] / 1000.0
background_estimate[2] = background_estimate[0] * 1000.0

estimate = {"norm_estimate": norm_estimate,
            "background_estimate": background_estimate}

# -----------------------------------------------------------------------------
# Analyze using MCMC
# -----------------------------------------------------------------------------
analysis = Do_MCMC(this).okgo(single_power_law_with_constant_not_normalized,
                              estimate=estimate,
                              iter=50000,
                              burn=10000,
                              thin=5,
                              progress_bar=True)


"""
z = Do_MCMC([ts0, ts1, ts2, ts3]).okgo(single_power_law, iter=50000, burn=10000, thin=5, progress_bar=False)

#z = Do_MCMC([ts]).okgo(single_power_law_with_constant, iter=50000, burn=10000, thin=5, progress_bar=False)
"""