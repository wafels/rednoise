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
import numpy as np
import astropy.units as u

import details_study as ds
import details_plots as dp
import details_simulated as dsim
import datalocationtools

from astropy.visualization.mpl_normalize import ImageNormalize
from astropy import visualization

from sunpy.cm import cm

from astropy.io import fits
import matplotlib.pyplot as plt

# Locations of the output datatypes
save_locations = ds.save_locations

# Identity of the data
ident = ds.ident

for wave in ds.waves:
    branches = [ds.corename, ds.original_datatype, wave]
    directory = datalocationtools.save_location_calculator({"aiadata": ds.dataroot}, branches)['aiadata']
    # Load in the derotated data into a datacube
    print('Acquiring data from ' + directory)
    filename = ds.source_filename(ds.study_type, wave)
    filepath = os.path.join(directory, filename)
    hdulist = fits.open(filepath)
    # The data needs to be re-ordered for use by the following analyses
    sda = np.swapaxes(hdulist[0].data, 0, 2)  # check this! 
    hdulist.close()

    #
    # Output the data in the format required
    #
    time_in_seconds = dsim.cadence.to(u.s).value * np.arange(0, sda.shape[2])

    # Let's simplify the output directory compared to what was done previously.
    # Since the output is intended to be that from a step1 process, let's
    # dump the output data into such a directory.

    # Locations of the output datatypes
    roots = {"project_data": os.path.join(ds.output_root, 'project_data'),
             "image": os.path.join(ds.output_root, 'image')}
    output_path = datalocationtools.save_location_calculator(roots, branches)['project_data']

    # Name the file
    output_filename = '{:s}_{:s}.step1.npz'.format(ds.study_type, wave)

    # Full filepath
    output_filepath = os.path.join(output_path, output_filename)

    # Make the subdirectory if it does not already exist
    if not os.path.exists(output_path):
        print('Creating {:s}'.format(output_path))
        os.makedirs(output_path)

    # Save the data
    print('Saving data to ' + output_filepath)
    np.savez(output_filepath, sda, time_in_seconds)




"""
if bradshaw_simulated_data:
    # Load the simulated data
    directory_listing = sorted(os.path.join(aia_data_location, f) for f in os.listdir(aia_data_location))
    list_of_data = []
    for f in directory_listing:
        if f[-5:] == '.fits':
            list_of_data.append(f)
        else:
            print('File that does not end in ".fits" detected, and not included in list = %s ' %f)
    print("Number of files = %i" % len(list_of_data))
    hdulist = fits.open(list_of_data[0])  # check this!
    # The data needs to be re-ordered for use by the followinf code
    sda = np.swapaxes(hdulist[0].data, 0, 2)  # check this!
    hdulist.close()
else:
    directory_listing = sorted(os.path.join(aia_data_location, f) for f in os.listdir(aia_data_location))
    data = np.load(directory_listing[0])
    sda = data['arr_0']
"""



""" Let's not bother with making a plot just yet.
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
"""
