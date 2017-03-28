#
# Create a simulated datacube time series
#
import os
import numpy as np
import astropy.units as u

from tools.timeseries_simulation import TimeSeriesFromModelSpectrum
from tools.rnspectralmodels4 import power_law_with_constant

file_path = os.path.expanduser('~/Data/ts/from_simulated_power_spectra/disk/sim0/94/from_simulated_power_spectra.npz')

# Datacube parameters
dt = (12.0 * u.s).to(u.s).value
nx = 30
ny = 40
nt = int(300 * (12 * u.hour).value)

# Oversampling parameters
v = 5
w = 7

# Power spectrum model parameters
psmp = [10.0, 1.4, 0.1]

ts = TimeSeriesFromModelSpectrum(power_law_with_constant, psmp,
                                 nt=nt, dt=dt, v=v, w=w)

sample_times = ts.dt * np.arange(0, ts.nt)

# Make the datacube
datacube = np.zeros((nx, ny, nt))
for i in range(0, nx):
    for j in range(0, ny):
        datacube[i, j, :] = np.real(ts.sample()[1])

# Save out the answer
np.savez(file_path, datacube, sample_times)
