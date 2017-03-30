#
# Create a simulated datacube time series
#
# Then run step0_convert_simulated_data.py
#
#
import os
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

from tools.timeseries_simulation import TimeSeriesFromModelSpectrum
from tools.rnspectralmodels4 import power_law_with_constant

file_path = os.path.expanduser('~/Data/ts/from_simulated_power_spectra/disk/sim0/94/from_simulated_power_spectra.npz')

# Datacube parameters
dt = (12.0 * u.s).to(u.s).value
nx = 10
ny = 10
nt = int(300 * (12 * u.hour).value)

# Oversampling parameters
v = 5
w = 7

# Power spectrum model parameters

psmp = [np.log(100), 4.0, np.log(0.00001)]

ts = TimeSeriesFromModelSpectrum(power_law_with_constant, psmp,
                                 nt=nt, dt=dt, v=v, w=w)

sample_times = ts.dt * np.arange(0, ts.nt)

"""
# Make the datacube
datacube = np.zeros((nx, ny, nt))
for i in range(0, nx):
    print('Row {:n} out of {:n}'.format(i, nx))
    for j in range(0, ny):
        datacube[i, j, :] = np.real(ts.sample()[1])

# Save out the answer
print('Saving data to {:s}.'.format(file_path))
np.savez(file_path, datacube, sample_times)
"""
# Plot a noisy spectrum and the true spectrum
d = np.real(ts.sample())
f = np.fft.fftfreq(ts.nt, ts.dt)
fpos = f > 0
p = np.abs(np.fft.fft(d))**2
plt.ion()
plt.loglog(f[fpos], p[fpos], label='noisy')
plt.loglog(f[fpos], ts.power_at_fourier_frequencies, label='true')
plt.xlabel('frequency')
plt.ylabel('FFT power')
plt.legend()
plt.title('Example noisy power spectrum')
