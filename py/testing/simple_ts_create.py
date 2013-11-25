"""
Look at the power spectrum of a time series constructed as follows:
(1) power law power spectrum
plus
(2) 

"""
 
import numpy as np
from rnsimulation import SimplePowerLawSpectrum, TimeSeriesFromPowerSpectrum
from timeseries import TimeSeries
from matplotlib import pyplot as plt
from rnfit2 import Do_MCMC
from pymcmodels import single_power_law_with_constant_not_normalized

# Set up the sample time and duration
dt = 12.0
nt = 300

# Over sampling of the red-noise time-series
V = 2
W = 3

# Power spectrum
pls = SimplePowerLawSpectrum([10.0, 2.0], nt=nt, dt=dt)

# Poseer spectrum time-series object
tsnew = TimeSeriesFromPowerSpectrum(pls, V=V, W=W, phase_noise=False, power_noise=False)

# Noise
noise_pls = SimplePowerLawSpectrum([10.0, 2.0], nt=nt, dt=dt)
rn = TimeSeriesFromPowerSpectrum(noise_pls, V=V, W=W)
noise = 0.0*rn.sample

# Create the simulated data
amplitude = 100
data = amplitude * tsnew.sample + noise

# Sample times
t = dt * np.arange(0, nt)

# Time series object
ts = TimeSeries(t, data)

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