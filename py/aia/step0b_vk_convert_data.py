"""
Step 0

Uses SunPy v0.9.4

Load in the VK data and convert it to a format for use with
the rest of the rednoise processing.

Here are some possible strategies for handling the simulated data.

(a) Full Monty

1. Load in the full 3d simulated data set.
2. Assume that the simulated data is observed at the Sun.  This means it has
a location on the Sun and the data rotates.
3. Differentially rotate each layer assuming a position on the Sun and an
observation time.
4. Save each layer as an individual FITS file.
5. Use step0_coalign_aia_data.py to proceed


(b) Dolly Dimple

1. Load in the VK data set.
2. Assume that the VK data has been de-rotated and co-aligned.
3. Save the data in the same format that is used by step2_create_power_spectra.

"""

import os
import pickle
import numpy as np
import pandas as pd
import astropy.units as u

import details_study as ds
import details_plots as dp
import details_simulated as dsim

from astropy.visualization.mpl_normalize import ImageNormalize
from astropy import visualization

from sunpy.cm import cm

from astropy.io import fits
import matplotlib.pyplot as plt


# Create the AIA source data location
aia_data_location = ds.aia_data_location["aiadata"]

# Extend the name if cross correlation is requested
extension = ds.aia_data_location

# Locations of the output datatypes
save_locations = ds.save_locations

# Identity of the data
ident = ds.ident

# Load in the derotated data into a datacube
print('Acquiring evenly sampled data from ' + aia_data_location)

# Load in the data
directory_listing = sorted(os.path.join(aia_data_location, f) for f in os.listdir(aia_data_location))
stop
info = np.load(directory_listing[0])

# Flip it round so it is in the correct format for later analysis
sda = np.swapaxes(info['data'], 0, 2)  # check this!
time_in_seconds = info['t']


# Output the data in the format required
times = {"date_obs": date_obs,
         "time_in_seconds": time_in_seconds}
#
# Step 2 has data shaped like (ny, nx, nt)
#
a = list()
a.append(ds.study_type)
a.append('disk')
a.append('sim0')
a.append('{:s}'.format(ds.wave))
z = '/home/ireland/ts/pickle/cc_True_dr_True_bcc_False/{:s}/{:s}/{:s}/{:s}/six_euv'.format(a[0], a[1], a[2], a[3])

filename = '{:s}_{:s}_{:s}_{:s}_six_euv.datacube.t0_None.pkl'.format(a[0], a[1], a[2], a[3])
if not os.path.exists(z):
    print('Creating {:s}'.format(z))
    os.makedirs(z)

pfilepath = '{:s}/{:s}'.format(z, filename)
print('Saving to {:s}'.format(pfilepath))
outputfile = open(pfilepath, 'wb')
pickle.dump(sda, outputfile)
pickle.dump(times, outputfile)
outputfile.close()

