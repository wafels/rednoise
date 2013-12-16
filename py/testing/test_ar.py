"""
Create a simple auto-regressive time series
"""

import numpy as np
from matplotlib import pyplot as plt
from timeseries import TimeSeries
from rnfit2 import Do_MCMC
from pymcmodels import single_power_law_with_constant_not_normalized

# Create some fake data
dt = 12.0
nt = 300

data = np.zeros(nt)

alpha = 0.0001

data[0] = 1.0
for i in range(0, nt - 1):
    data[i+1] = data[i] + alpha*np.random.normal()


ts = TimeSeries(dt * np.arange(0, nt), data)

plt.figure(1)
ts.peek_ps()
plt.loglog()


plt.figure(2)
ts.peek()

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

z = Do_MCMC(this).okgo(single_power_law_with_constant_not_normalized, estimate=estimate,
                       iter=50000, burn=10000, thin=5, progress_bar=True)
