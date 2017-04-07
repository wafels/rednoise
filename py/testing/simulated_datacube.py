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


class DataCube:
    def __init__(self, dt, nt, nx, ny, model, a, v=1, w=1, constant=1.0):
        self.dt = dt
        self.nt = nt
        self.nx = nx
        self.ny = ny
        self.model = model
        self.a = a
        self.v = v
        self.w = w
        self.constant = constant

cubes = dict()
ncube = 0
for hour_duration in (6.0, 12.0):
    for index in (1.0, 1.5, 2.0, 2.5, 3.0):
        ncube += 1
        cube = DataCube((12.0 * u.s).to(u.s).value,
                        int(300 * (hour_duration * u.hour).value),
                        100,
                        100,
                        power_law_with_constant,
                        [np.log(100), index, np.log(0.1)])

        # Store all the cubes
        cubes[ncube] = cube

for choice in range(1, ncube + 1):

    # Select a cube
    this_cube = cubes[choice]

    # Datacube parameters
    dt = this_cube.dt
    nx = this_cube.nx
    ny = this_cube.ny
    nt = this_cube.nt

    # Oversampling parameters
    v = this_cube.v
    w = this_cube.w

    # Power spectrum model parameters
    a = this_cube.a
    model = this_cube.model

    # Create the time series object
    ts = TimeSeriesFromModelSpectrum(model, a, nt=nt, dt=dt, v=v, w=w)
    sample_times = ts.dt * np.arange(0, ts.nt)
    constant = this_cube.constant

    # Make the datacube
    datacube = np.zeros((nx, ny, nt))
    for i in range(0, nx):
        for j in range(0, ny):
            datacube[i, j, :] = constant + np.real(ts.sample())

    this_dir = os.path.expanduser('~/Data/ts/from_simulated_power_spectra_{:n}/disk/sim0/335/'.format(choice))
    if not os.path.isdir(this_dir):
        os.makedirs(this_dir)
    file_path = os.path.expanduser('~/Data/ts/from_simulated_power_spectra_{:n}/disk/sim0/335/from_simulated_power_spectra_{:n}.npz'.format(choice, choice))

    # Save out the answer
    print('Saving data to {:s}.'.format(file_path))
    np.savez(file_path, datacube, sample_times)

# Plot the over sampled power and the power at the input Fourier frequencies
plt.figure(1)
plt.clf()
plt.ion()
plt.loglog(ts.pos_osf, ts.over_sampled_power,
           label='v={:n}, w={:n}'.format(v, w))
plt.loglog(ts.pos_iff, ts.power_at_fourier_frequencies, label='Fourier set')
plt.legend()
plt.xlabel('frequency')
plt.ylabel('FFT power')
plt.title('Power spectrum at original and oversampled FFT frequencies')
plt.grid()
plt.show()

# Plot a noisy spectrum and the true spectrum
plt.figure(2)
plt.clf()
d = np.real(ts.sample())
f = np.fft.fftfreq(ts.nt, ts.dt)
fpos = f > 0
p = np.abs(np.fft.fft(d)) ** 2
plt.loglog(f[fpos], p[fpos] / np.max(p[fpos]), label='noisy')
plt.loglog(f[fpos], ts.power_at_fourier_frequencies / np.max(
    ts.power_at_fourier_frequencies), label='true')
plt.xlabel('frequency')
plt.ylabel('FFT power')
plt.legend()
plt.title('Example noisy power spectrum')
plt.grid()

# Plot a sample time series
plt.figure(3)
plt.clf()
plt.plot(sample_times, datacube[0, 0, :])
plt.xlabel('time')
plt.ylabel('amplitude')
plt.title('Example noisy time series\nnt={:n}, dt={:n}'.format(nt, dt))
plt.grid()
