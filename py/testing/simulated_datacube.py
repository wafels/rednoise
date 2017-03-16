#
# Create a simulated datacube time series
#
import numpy as np
import astropy.units as u

from tools.timeseries_simulation import TimeSeriesFromModelSpectrum
from tools import rnspectralmodels4

# Datacube parameters
dt = (12.0 * u.s).to(u.s).value
nx = 3
ny = 3
nt = 300 * 12

# Oversampling parameters
v = 5
w = 7

# Power spectrum model parameters
model = rnspectralmodels4.broken_power_law_with_constant
psmp = []

ts = TimeSeriesFromModelSpectrum(broken_power_law_with_constant, psmp,
                                 nt=nt, dt=dt, v=v, w=w)

# Make the datacube
datacube = np.zeros((nx, ny, nt))
for i in range(0, nx):
    for j in range(0, ny):
        datacube[i, j, :] = ts.sample()[1]

# Save out the answer
