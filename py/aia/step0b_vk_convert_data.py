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
from astropy.time import Time
import details_study as ds
import details_plots as dp
import details_simulated as dsim

from astropy.visualization.mpl_normalize import ImageNormalize
from astropy import visualization

from sunpy.map import Map

from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord
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
info = np.load(directory_listing[1])

# Flip it round so it is in the correct format for later analysis
sda = np.swapaxes(np.swapaxes(info['data'], 0, 2), 0, 1)  # check this!
time_in_seconds = info['t']

# Load in the initial time
f = open(directory_listing[0])
date_obs = Time(f.read())
f.close()

# Output the data in the format required
times = {"date_obs": date_obs,
         "time_in_seconds": time_in_seconds}

# branch location
b = [ds.corename, ds.sunlocation, ds.fits_level, ds.wave, 'six_euv']

# Region identifier name
region_id = ds.datalocationtools.ident_creator(b)

# Output location
output = ds.datalocationtools.save_location_calculator(ds.roots, b)["pickle"]

# Output filename
ofilename = os.path.join(output,
                         region_id + '.datacube.{:s}'.format(ds.index_string))

if not os.path.exists(output):
    print('Creating {:s}'.format(output))
    os.makedirs(output)

ofilename = '{:s}.{:s}'.format(ofilename, 'pkl')
print('Saving to {:s}'.format(ofilename))
f = open(ofilename, 'wb')
pickle.dump(sda, f)
pickle.dump(times, f)
f.close()

# Create maps of the data for display purposes
fake_map = ds.make_simple_map(ds.wave, date_obs, np.mean(sda, 2))
plt.close('all')
fake_map.peek()
# Output location
output = ds.datalocationtools.save_location_calculator(ds.roots, b)["image"]

# Output filename
filepath = os.path.join(output, region_id + '.summary_image.{:s}.png'.format(ds.index_string))
print('Saving to ' + filepath)
plt.savefig(filepath, bbox_inches='tight')
plt.close('all')


# Create maps of the data for display purposes
fake_map = ds.make_simple_map(ds.wave, date_obs, sda[:, :, 0])
plt.close('all')
figure = plt.figure(4)
ax = plt.subplot(projection=fake_map)

# Create the composite map

box = fake_map.pixel_to_world((ds.ar_x[0].value, ds.ar_x[0].value, ds.ar_x[1].value, ds.ar_x[1].value)*u.pix,
                              (ds.ar_y[0].value, ds.ar_y[1].value, ds.ar_y[0].value, ds.ar_y[1].value)*u.pix)

fake_map.plot(axes=ax)
ax.plot_coord(box, color='c')
plt.colorbar()
plt.show()


fake_map.plot_coord(box)
# Add a line that indicates where the best Long score is

# Output location
output = ds.datalocationtools.save_location_calculator(ds.roots, b)["image"]

# Output filename
filepath = os.path.join(output, region_id + '.example_image.{:s}.png'.format(ds.index_string))
print('Saving to ' + filepath)
plt.savefig(filepath, bbox_inches='tight')
plt.close('all')

