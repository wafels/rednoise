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
sda = hdulist[0].data  # check this!
hdulist.close()

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
pickle.dump(np.swapaxes(sda, 0, 2), outputfile)
pickle.dump(times, outputfile)
outputfile.close()

stretch = {'papern_bradshaw_simulation_high_fn': 0.00001,
           'papern_bradshaw_simulation_intermediate_fn': 0.001,
           'papern_bradshaw_simulation_low_fn': 0.001}

#stretch = {'papern_bradshaw_simulation_high_fn': 0.001,
#           'papern_bradshaw_simulation_intermediate_fn': 0.001,
#           'papern_bradshaw_simulation_low_fn': 0.001}

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
if dsim.method == 'simple':
    pass

#
# Start manipulating the data
#
print("Loading data")
mc = Map(list_of_data, cube=True)

# Get the date and times from the original mapcube
date_obs = []
time_in_seconds = []
for m in mc:
    date_obs.append(parse_time(m.date))
    time_in_seconds.append((date_obs[-1] - date_obs[0]).total_seconds())
times = {"date_obs": date_obs, "time_in_seconds": np.asarray(time_in_seconds)}


# Solar de-rotation and cross-correlation operations will be performed relative
# to the map at this index.
layer_index = len(mc) / 2
t_since_layer_index = times["time_in_seconds"] - times["time_in_seconds"][layer_index]
filepath = os.path.join(save_locations['image'], ident + '.cross_correlation.png')
#
# Apply solar derotation
#
if ds.derotate:
    print("\nPerforming de-rotation")

    # Calculate the solar rotation of the mapcube
    print("Calculating the solar rotation shifts")
    sr_shifts = calculate_solar_rotate_shift(mc, layer_index=layer_index)

    # Plot out the solar rotation shifts
    filepath = os.path.join(save_locations['image'], ident + '.solar_derotation.png')
    step0_plots.plot_shifts(sr_shifts, 'shifts due to solar de-rotation',
                            layer_index, filepath=filepath)
    filepath = os.path.join(save_locations['image'], ident + '.time.solar_derotation.png')
    step0_plots.plot_shifts(sr_shifts, 'shifts due to solar de-rotation',
                            layer_index, filepath=filepath,
                            x=t_since_layer_index, xlabel='time relative to reference layer (s)')

    # Apply the solar rotation shifts
    print("Applying solar rotation shifts")
    data = mapcube_solar_derotate(mc,
                                  layer_index=layer_index, shift=sr_shifts, clip=True,
                                  order=1)
else:
    data = Map(list_of_data, cube=True)

#
# Coalign images by cross correlation
#
if ds.cross_correlate:
    if use_base_cross_correlation_channel:
        ccbranches = [ds.corename, ds.sunlocation, ds.fits_level, ds.base_cross_correlation_channel]
        ccsave_locations = ds.datalocationtools.save_location_calculator(ds.roots, ccbranches)
        ccident = ds.datalocationtools.ident_creator(ccbranches)
        ccfilepath = os.path.join(ccsave_locations['pickle'], ccident + '.cross_correlation.pkl')

        if ds.wave == ds.base_cross_correlation_channel:
            print("\nPerforming cross_correlation and image shifting")
            cc_shifts = calculate_match_template_shift(mc, layer_index=layer_index)
            print("Saving cross correlation shifts to %s" % filepath)
            f = open(ccfilepath, "wb")
            pickle.dump(cc_shifts, f)
            f.close()

            # Now apply the shifts
            data = mapcube_coalign_by_match_template(data, layer_index=layer_index, shift=cc_shifts)
        else:
            print("\nUsing base cross-correlation channel information.")
            print("Loading in shifts to due cross-correlation from %s" % ccfilepath)
            f = open(ccfilepath, "rb")
            cc_shifts = pickle.load(f)
            f.close()
            print("Shifting images")
            data = mapcube_coalign_by_match_template(data, layer_index=layer_index, shift=cc_shifts)
    else:
        print("\nCalculating cross_correlations.")
        #
        # The 131 data has some very significant shifts that may be
        # related to large changes in the intensity in small portions of the
        # data, i.e. flares.  This may be throwing the fits off.  Perhaps
        # better to apply something like a log?
        #
        if ds.wave == '131' or (ds.wave == '171' and ds.study_type == 'paper3_BLSGSM'):
            cc_func = np.sqrt
        else:
            cc_func = _default_fmap_function
        print('Data will have %s applied to it.' % cc_func.__name__)
        cc_shifts = calculate_match_template_shift(data, layer_index=layer_index, func=cc_func)
        print("Applying cross-correlation shifts to the data.")
        data = mapcube_coalign_by_match_template(data, layer_index=layer_index, shift=cc_shifts)

    # Plot out the cross correlation shifts
    filepath = os.path.join(save_locations['image'], ident + '.cross_correlation.png')
    step0_plots.plot_shifts(cc_shifts, 'shifts due to cross correlation \n using %s' % cc_func.__name__,
                            layer_index, filepath=filepath)

    filepath = os.path.join(save_locations['image'], ident + '.time.cross_correlation.png')
    step0_plots.plot_shifts(cc_shifts, 'shifts due to cross correlation \n using %s'  % cc_func.__name__,
                            layer_index, filepath=filepath,
                            x=t_since_layer_index, xlabel='time relative to reference layer (s)')
#
# Save the full dataset
#
directory = save_locations['pickle']
filename = ident + '.full_mapcube{:s}{:s}pkl'.format(ds.step0_output_information, '_ireland=step0.')
pfilepath = os.path.join(directory, filename)
print('Saving data to ' + pfilepath)
outputfile = open(pfilepath, 'wb')
pickle.dump(data, outputfile)
pickle.dump(layer_index, outputfile)
outputfile.close()
"""
