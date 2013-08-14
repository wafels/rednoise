"""
Co-align a set of maps
"""

# Test 1
import sunpy
import scipy
import numpy as np
from matplotlib import pyplot as plt
import pymcmodels
import pymc
import rnspectralmodels

directory = '/home/ireland/Data/AIA_Data/SOL2011-04-30T21-45-49L061C108/171/'

difflimit = 0.05

maps = sunpy.Map(directory, cube=True)

ref_index = 0
ref_center = maps[ref_index].center
ref_time = maps[ref_index].date


nt = len(maps[:])
ny = maps[0].shape[0]
nx = maps[0].shape[1]

datacube = np.zeros((ny, nx, nt))

max_abs_xdiff = 0.0
max_abs_ydiff = 0.0

for t, m in enumerate(maps):
    # Assume all the maps have the same pointing and the Sun rotates in the
    # field of view.  The center of the field of view at the start time gives
    # the initial field of view that then moves at later times.  This initial
    # location has to be rotated forward to get the correct center of the FOV
    # for the next maps.
    newx, newy = sunpy.coords.rot_hpc(ref_center['x'],
                                      ref_center['y'],
                                      ref_time,
                                      m.date)
    if newx is None:
        xdiff = 0
    else:
        xdiff = newx - ref_center['x']
    if newy is None:
        ydiff = 0
    else:
        ydiff = newy - ref_center['y']

    if np.abs(xdiff) > max_abs_xdiff:
        max_abs_xdiff = np.ceil(np.abs(xdiff))

    if np.abs(ydiff) > max_abs_ydiff:
        max_abs_ydiff = np.ceil(np.abs(ydiff))

    # In matplotlib, the origin of the image is in the top right of the
    # the array.  Therefore, shifting in the y direction has to be reversed.
    # Also, the first dimension in the numpy array refers to what a solar
    # physicist refers to as the north-south (i.e., 'y') direction.
    #print t, xdiff, ydiff
    if np.abs(ydiff) >= difflimit and np.abs(xdiff) >= difflimit:
        shifted = scipy.ndimage.interpolation.shift(m.data, [-ydiff, xdiff])
    else:
        shifted = m.data
    # Store all the shifted data.
    datacube[..., t] = shifted
    #print t, m.date, maps[t].center, xdiff, ydiff

# Zero out the edges where the data has been moved
dc = datacube[max_abs_ydiff + 1:, max_abs_xdiff + 1:, :]


### Start the power law analysis

iterations = 50000
burn = iterations / 4
thin = 10


ny = dc.shape[0]
nx = dc.shape[1]
nt = dc.shape[2]

# FFT frequencies
f = np.fft.fftfreq(nt, 12.0)
# positive frequencies
fpos_index = f > 0
fpos = f[fpos_index]

# Set up the storage array
nbins = 100
indices = np.zeros((ny, nx, nbins))
index_range = [-0.5, 4.0]

normalization = np.zeros((ny, nx, nbins))
norm_range = [-10.0, 40.0]

background = np.zeros((ny, nx, nbins))
back_range = [-10.0, 40.0]

i = 100
j = 100
ts = dc[i, j, :]
pwr = ((np.abs(np.fft.fft(ts))) ** 2)[fpos_index]
pwr_law_with_constant = pymcmodels.single_power_law_with_constant(fpos, pwr)
# Set up the MCMC model
M1 = pymc.MCMC(pwr_law_with_constant)

# Run the sampler
M1.sample(iter=iterations, burn=burn, thin=thin, progress_bar=False)
print np.min(M1.trace("power_law_index")[:]), np.max(M1.trace("power_law_index")[:])
print np.min(M1.trace("power_law_norm")[:]), np.max(M1.trace("power_law_norm")[:])
print np.min(M1.trace("background")[:]), np.max(M1.trace("background")[:])

# Get the results for each variable
indices[i, j, :] = np.histogram(M1.trace("power_law_index")[:], range=index_range, bins=nbins)[0]
normalization[i, j, :] = np.histogram(M1.trace("power_law_norm")[:], range=norm_range, bins=nbins)[0]
background[i, j, :] = np.histogram(M1.trace("background")[:], range=back_range, bins=nbins)[0]

#plt.figure(1)
#plt.loglog(fpos, rnspectralmodels.power_law_with_constant(fpos, [np.mean(M1.trace("power_law_norm")[:]),
#                                                                 np.mean(M1.trace("power_law_index")[:]),
#                                                                 np.mean(M1.trace("background")[:])]))

# Save the results


# Plot some histograms of the results for the entire data cube

# Probability histogram of the measured power law index

# Probability histogram of the measured constant background

# Probability histogram of the normalization
