"""
Test the transfer function

"""

from matplotlib import pyplot as plt
import numpy as np
from timeseries import TimeSeries
import tsutils
plt.ion()

dt = 12.0
nt = 18000
period = 300.0
window = 31
np.random.seed(seed=1)

t = dt * np.arange(0, nt)

noise = np.random.normal(size=nt)

amplitude = 0.3
data1 = amplitude * np.sin(2 * np.pi * t / period) + noise

data2 = tsutils.movingaverage(data1, window)

ts1 = TimeSeries(t, data1)

ts2 = TimeSeries(t, data2)

ts3 = TimeSeries(t, data1 - data2)

window = window / 2
w1 = 1.0 / (2 * window + 1)
index2 = dt * (np.arange(0, 2 * window + 1) - window)
weight2 = np.zeros_like(index2) + w1
angfreq = 2 * np.pi * ts1.pfreq

transfer_function12 = tsutils.transfer_function(index2, weight2, angfreq)
"""
plt.figure(1)
ts1.peek_ps()
ts2.peek_ps()

plt.figure(2)
plt.plot(ts1.pfreq, transfer_function12)

plt.figure(3)
plt.plot(ts1.pfreq, ts2.PowerSpectrum.ppower / ts1.PowerSpectrum.ppower)
"""
index3 = dt * (np.arange(0, 2 * window + 1) - window)
weight3 = np.zeros_like(index3) - w1
weight3[index3 == 0] = 1.0 - w1
angfreq = 2 * np.pi * ts1.pfreq

transfer_function13 = tsutils.transfer_function(index3, weight3, angfreq)

plt.figure(4)
plt.plot(ts1.pfreq, ts3.PowerSpectrum.ppower / ts1.PowerSpectrum.ppower, label='observed')
plt.plot(ts1.pfreq, transfer_function13, label='theoretical')
plt.legend()

plt.figure(6)
ts1.peek_ps()
ts3.peek_ps()

