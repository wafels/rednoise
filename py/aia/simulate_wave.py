#
# Program to simulate a time distance plot as used in coronal
# seismology
#
import numpy as np
from rnsimulation import SimplePowerLawSpectrumWithConstantBackground, TimeSeriesFromPowerSpectrum
import matplotlib.pyplot as plt
from timeseries import TimeSeries
from paper1 import log_10_product

nt = 60 * 60 / 12
nx = 200
dt = 12.0

data = np.zeros((nx, nt))

t = dt * np.arange(0, nt)

k = 0.1
w = 0.003333

#np.random.seed(seed=1)

# Put a rednoise signal in every time bin
for i in range(0, nx):
    ind = np.random.uniform(low=1.0, high=3.0)
    norm = np.random.uniform(low=1.0, high=10.0)
    back = np.random.uniform(low=-10.0, high=-3.0)
    pls1 = SimplePowerLawSpectrumWithConstantBackground([10.0, ind, -5.0],
                                                    nt=nt,
                                                    dt=dt)
    data[i, :] = TimeSeriesFromPowerSpectrum(pls1).sample
    data[i, :] = data[i, :] + 0.005 * np.sin(2 * np.pi * (k * i - w * t))

data = data / data.max()
#
ts = TimeSeries(t, data[10, :])

freqs = ts.PowerSpectrum.frequencies.positive
example = ts.ppower

plt.figure(1)
plt.imshow(data, origin='lower', cmap='Greys', aspect='auto')
plt.xlabel('time (arbitrary units)')
plt.ylabel('distance along loop (arbitrary units')
plt.show()

ax = plt.subplot(111)

# Set the scale type on each axis
ax.set_xscale('log')
ax.set_yscale('log')

# Set the formatting of the tick labels

xformatter = plt.FuncFormatter(log_10_product)
ax.xaxis.set_major_formatter(xformatter)

#yformatter = plt.FuncFormatter(log_10_product)
#ax.yaxis.set_major_formatter(yformatter)

# Geometric mean
ax.plot(1000 * freqs, example)

plt.xlabel('frequency (%s)' % ('mHz'))
plt.ylabel('power (arb. units)')
plt.title('?')
plt.show()

