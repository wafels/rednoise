"""
Step 1

Load in the de-rotated and co-aligned map cube, and make some cutouts that
focus on specific regions of interest.
"""

import os
from copy import deepcopy
import pickle

from sunpy.time import parse_time
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.io import fits

import details_study as ds
import analysis_get_data

# Load in the derotated data into a datacube
directory = ds.save_locations['pickle']
filename = ds.ident + '.full_mapcube{:s}.pkl'.format(ds.processing_info)
print('Acquiring mapcube from ' + ds.save_locations['pickle'])
print('Filename = ' + filename)
outputfile = open(os.path.join(directory, filename), 'rb')
mc = pickle.load(outputfile)
layer_index = pickle.load(outputfile)
outputfile.close()

# Get the date and times from the original mapcube
date_obs = []
time_in_seconds = []
for m in mc:
    date_obs.append(parse_time(m.date))
    time_in_seconds.append((date_obs[-1] - date_obs[0]).total_seconds())
times = {"date_obs": date_obs, "time_in_seconds": np.asarray(time_in_seconds)}

# Layer on which all de-rotations, etc are based on.
mc_layer = mc[layer_index]


#
# A function that calculates some necessary region information in order to
# for us to extract a subcube of data.
#
def calculate_region_information(regions):
    # Rectangular patches
    for region in regions:
        # Next region
        R = regions[region]

        # Define a matplotlib rectangular patch to show the region on a map
        new_rectangle = Rectangle((R['llx'].value, R['lly'].value),
                                  R['width'].value, R['height'].value,
                                  label=region, fill=False,
                                  facecolor='b', edgecolor='r', linewidth=2)

        # Store the information about the rectangle
        R["patch"] = new_rectangle
        R["path"] = new_rectangle.get_path().transformed(transform=new_rectangle.get_transform())
        R['label_offset'] = {"x": 0.0, "y": -10}

        # Information about the ranges
        R['xrange'] = [R['llx'].value, R['llx'].value + R['width'].value]
        R['yrange'] = [R['lly'].value, R['lly'].value + R['height'].value]
        llxy_pixel = mc_layer.data_to_pixel(R['llx'], R['lly'])
        R['llxy_pixel'] = [np.int(np.floor(llxy_pixel[0].value)), np.int(np.floor(llxy_pixel[1].value))]

        # Small changes in the plate scale in each AIA channel can mean that
        # the height and widths of each region depend on the data. To mitigate
        # against this we fix the size of the plate scale.
        fixed_aia_scale = ds.fixed_aia_scale#{'x': 0.6, 'y': 0.6}
        R['width_pixel'] = np.int(np.floor(R['width'] / fixed_aia_scale['x']).value)
        R['height_pixel'] = np.int(np.floor(R['height'] / fixed_aia_scale['y']).value)
    return regions
#
# Create some time-series for further analysis
#
if ds.sunlocation == 'disk' or ds.sunlocation == 'debug':
    #
    # Define the number of regions and their extents.  Use Helio-projective
    # co-ordinates.  These co-ordinates should be chosen in reference to the
    # time of the reference layer (layer_index).
    #
    regions_mpl = calculate_region_information(deepcopy(ds.regions))

    for region in regions_mpl:
        # Next region
        R = regions_mpl[region]

        # The mapcube images a piece of Sun that inevitably moves.  Defining
        # the subcube using HPC co-ords will not work in this case.  Therefore
        # we need to convert the HPC co-ords into array indices and return
        # numpy arrays.  Create the subcube
        range_x = [R['llxy_pixel'][1], R['llxy_pixel'][1] + R['height_pixel']]
        range_y = [R['llxy_pixel'][0], R['llxy_pixel'][0] + R['width_pixel']]
        subdata = (mc.as_array())[range_x[0]: range_x[1], range_y[0]: range_y[1], :]

        print('\nSub datacube (size %i, %i, %i)' % subdata.shape)

        # Create a SunPy map that describes the location of the data
        region_submap = mc_layer.submap(range_y * u.pix, range_x * u.pix)

        # Region identifier name
        region_id = ds.ident + '_' + region

        # branch location
        b = [ds.corename, ds.sunlocation, ds.fits_level, ds.wave, region]

        # Output location
        output = ds.datalocationtools.save_location_calculator(ds.roots, b)["pickle"]

        # Output filename
        ofilename = os.path.join(output, region_id + '.datacube{:s}.pkl'.format(ds.processing_info))

        # Open the file and write out the data we need for step 2
        outputfile = open(ofilename, 'wb')

        # Write out the cube of data we want
        pickle.dump(subdata, outputfile)

        # Write out the times when each layer was taken
        pickle.dump(times, outputfile)

        # Write out the layer at which the rotation etc refer too
        pickle.dump(mc_layer, outputfile)

        # Write out the region information
        pickle.dump(R, outputfile)

        # Write out the regional submap
        pickle.dump(region_submap, outputfile)

        # Close the data
        outputfile.close()
        print('Saved to ' + ofilename)

        # Write out a FITS file
        hdu = fits.PrimaryHDU(subdata)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(ofilename + '.fits', clobber=True)

        #
        # Download the required feature/event data
        #
        root_filename = region_id + '.fevent{:s}.pkl'.format(ds.processing_info)
        for region in ds.regions:
            # Next region
            R = ds.regions[region]
            for fevent in ds.fevents:
                filename = region_id + '.{:s}.{:s}.fevent{:s}.pkl'.format(fevent[0], fevent[1], ds.processing_info)
                fevents = analysis_get_data.fevent_outline([mc[0].date, mc[-1].date], R,
                                                           mc_layer.date,
                                                           fevent=fevent,
                                                           download=True,
                                                           directory=output,
                                                           filename=filename)

#
# Plot where the regions and any feature and events are
#
plt.ion()
plt.close('all')
fig, ax = plt.subplots()
ret = mc_layer.plot()
filepath = os.path.join(ds.save_locations['image'], ds.ident + '.regions{:s}'.format(ds.processing_info))
for region in regions_mpl.keys():
    patch = regions_mpl[region]["patch"]
    label_offset = regions_mpl[region]["label_offset"]
    ax.add_patch(patch)
    llxy = patch.get_xy()
    height = patch.get_height()
    width = patch.get_width()
    plt.text(llxy[0] + width + label_offset['x'],
             llxy[1] + height + label_offset['y'],
             patch.get_label(),
             bbox=dict(facecolor='w', alpha=0.5))

# Plot the features and events
dt = (30 * u.day).to(u.s).value
for i, fevent in enumerate(fevents):
    this_dt = np.abs((parse_time(fevent.time) - mc_layer.date).total_seconds())
    if this_dt < dt:
        this_fevent = i
ax.add_artist(fevents[this_fevent].solar_rotate(mc_layer.date).mpl_polygon)
ax.autoscale_view()
plt.show()
