from __future__ import absolute_import
"""
Co-align a set of maps and make sure the
"""

# Test 1
import scipy
import numpy as np
from matplotlib import pyplot as plt
import pymcmodels
import pymc
from cubetools import derotated_datacube_from_mapcube
import sunpy
import pickle

# Directory where the data is
wave = '193'
directory = '/home/ireland/Data/AIA_Data/SOL2011-04-30T21-45-49L061C108/' + wave + '/'
print('Loading ' + directory)
# Get a mapcube
maps = sunpy.Map(directory, cube=True)

# Get the datacube
dc = derotated_datacube_from_mapcube(maps)
ny = dc.shape[0]
nx = dc.shape[1]
nt = dc.shape[2]


def do_mcmc(ts, fpos_index, fpos):
    # Calculate the power at the positive frequencies
    pwr = ((np.abs(np.fft.fft(ts))) ** 2)[fpos_index]
    # Normalize the power
    pwr = pwr / pwr[0]
    # Get the PyMC model
    pwr_law_with_constant = pymcmodels.single_power_law_with_constant(fpos, pwr)
    # Set up the MCMC model
    M = pymc.MCMC(pwr_law_with_constant)
    # Do the MAP calculation
    M.sample(iter=50000, burn=10000, thin=10, progress_bar=False)
    return M


def visualize(datacube, delay=0.1, range=None):
    """
    Visualizes a datacube.  Requires matplotlib 1.1
    """
    nt = datacube.shape[2]
    print nt
    fig = plt.figure()

    axes = fig.add_subplot(111)
    axes.set_xlabel('X-position')
    axes.set_ylabel('Y-position')

    #extent = wave_maps[0].xrange + wave_maps[0].yrange
    #axes.set_title("%s %s" % (wave_maps[0].name, wave_maps[0].date))

    img = axes.imshow(dc[:, :, 0], origin='lower')
    #fig.colorbar(img)
    fig.show()

    for z in np.arange(1, nt - 1):
        m = datacube[:, :, z]
        #axes.set_title("%i" % (i))
        img.set_data(m)
        plt.pause(delay)
    return None

#visualize(dc)

# FFT frequencies
f = np.fft.fftfreq(nt, 12.0)
# positive frequencies
fpos_index = f > 0
fpos = f[fpos_index]

# Set up the storage array
bins = 100
results = np.zeros((3, ny, nx, bins))
pli_range = [-1.0, 6.0]

nor_range = [-10, 10]

bac_range = [-20, 20]

for i in range(0, ny):
    print i, ny
    for j in range(0, nx):
        # Get the time series
        ts = dc[i, j, :]

        # Do the MCMC
        answer = do_mcmc(ts, fpos_index, fpos)

        q1 = answer.trace("power_law_index")[:]
        results[0, i, j, :] = np.histogram(q1, bins=bins, range=pli_range)[0]

        q2 = answer.trace("power_law_norm")[:]
        results[1, i, j, :] = np.histogram(q2, bins=bins, range=nor_range)[0]

        q3 = answer.trace("background")[:]
        results[2, i, j, :] = np.histogram(q3, bins=bins, range=bac_range)[0]

epli = np.histogram(q1, bins=bins, range=pli_range)[1]
epln = np.histogram(q2, bins=bins, range=nor_range)[1]
ebac = np.histogram(q3, bins=bins, range=bac_range)[1]

# Save the results

fname = wave + '.data.pkl'
print 'Saving to ' + fname
output = open(fname, 'wb')
pickle.dump(results, output)
pickle.dump(epli, output)
pickle.dump(epln, output)
pickle.dump(ebac, output)
output.close()

#plt.figure(1)
#plt.loglog(fpos, rnspectralmodels.power_law_with_constant(fpos, [np.mean(M1.trace("power_law_norm")[:]),
#                                                                 np.mean(M1.trace("power_law_index")[:]),
#                                                                 np.mean(M1.trace("background")[:])]))

# Save the results


# Plot some histograms of the results for the entire data cube

# Probability histogram of the measured power law index

# Probability histogram of the measured constant background

# Probability histogram of the normalization
