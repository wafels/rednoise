"""
Step 0

Load in the Bradshaw simulated data and convert it to a format for use with
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

1. Load in the full 3d simulated data set.
2. Assume that the simulated data represents real observational data that has
been de-rotated and co-aligned.
3. Save the data in the same format that is used by step2_create_power_spectra.

"""

import os
from datetime import timedelta
import pickle

import numpy as np

import astropy.units as u

from sunpy.time import parse_time
from sunpy.map import Map
from sunpy.image.coalignment import mapcube_coalign_by_match_template, calculate_match_template_shift, _default_fmap_function
from sunpy.physics.transforms.solar_rotation import mapcube_solar_derotate, calculate_solar_rotate_shift
import step0_plots
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
print('Acquiring data from ' + aia_data_location)

# Bradshaw simulated data?
bradshaw_simulated_data = ds.study_type in ('papern_bradshaw_simulation_low_fn',
                                            'papern_bradshaw_simulation_intermediate_fn',
                                            'papern_bradshaw_simulation_high_fn')

if bradshaw_simulated_data:
    # Get the simulated data
    directory_listing = sorted(os.path.join(aia_data_location, f) for f in os.listdir(aia_data_location))
    list_of_data = []
    for f in directory_listing:
        if f[-5:] == '.fits':
            list_of_data.append(f)
        else:
            print('File that does not end in ".fits" detected, and not included in list = %s ' %f)
    print("Number of files = %i" % len(list_of_data))
    hdulist = fits.open(list_of_data[0])  # check this!
    sda = np.swapaxes(hdulist[0].data, 0, 2)  # check this!
    hdulist.close()

else:
    directory_listing = sorted(os.path.join(aia_data_location, f) for f in os.listdir(aia_data_location))
    data = np.load(directory_listing)


#
# Output the data in the format required
#
times = {"date_obs": "2016-08-15 01:23:45", "time_in_seconds": dsim.cadence.to(u.s).value * np.arange(0, sda.shape[2])}
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

if bradshaw_simulated_data:
    # Color stretching for the Bradshaw simulated data
    stretch = {'papern_bradshaw_simulation_high_fn': 0.00001,
               'papern_bradshaw_simulation_intermediate_fn': 0.001,
               'papern_bradshaw_simulation_low_fn': 0.001}

    nt = sda.shape[0]
    im = sda[nt//2, :, :]
    cmap = cm.sdoaia171  # get_cmap(self._get_cmap_name())
    norm = ImageNormalize(stretch=visualization.AsinhStretch(stretch[ds.corename]))

    plt.close('all')
    plt.imshow(im, cmap=cmap, norm=norm, origin='bottom')
    plt.xlabel('x (pixels)', fontsize=dp.fontsize)
    plt.ylabel('y (pixels)', fontsize=dp.fontsize)
    title = ds.sim_name[ds.corename]
    title += '\nsimulated AIA 171 Angstrom emission'
    plt.title(title, fontsize=dp.fontsize)
    #plt.colorbar(label='emission')
    plt.savefig('/home/ireland/Desktop/emission.{:s}.png'.format(ds.sim_name[ds.corename]), bbox_inches='tight')
    plt.close('all')
