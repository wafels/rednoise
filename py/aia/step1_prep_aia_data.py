"""
Load in the de-rotated and co-aligned data cube, and make some
cutouts.
"""

import os

import cPickle as pickle

from sunpy.time import parse_time
import numpy as np
from matplotlib.patches import Rectangle

import study_details as sd
import step1_plots

# Load in the derotated data into a datacube
directory = sd.save_locations['pickle']
filename = sd.ident + '.full_mapcube.pkl'
print('Acquiring mapcube from ' + sd.save_locations['pickle'])
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
# Create some time-series for further analysis
#
if sd.sunlocation == 'disk':
    #
    # Define the number of regions and their extents.  Use Helio-projective
    # co-ordinates
    #
    regions = {"sunspot": {"llx": -150.0, "lly": -350, "width": 20, "height": 15},
               "loop footpoints": {"llx": -100, "lly": -355, "width": 22, "height": 24},
               "quiet Sun": {"llx": -50, "lly": -360, "width": 15, "height": 26},
               "active region": {"llx": 0, "lly": -370, "width": 9, "height": 27}}

    # Rectangular patches
    for region in regions:
        # Next region
        R = regions[region]

        # Define a matplotlib rectangular patch to show the region on a map
        new_rectangle = Rectangle((R['llx'], R['lly']),
                                  R['width'], R['width'],
                                  label=region, fill=False,
                                  facecolor='b', edgecolor='r', linewidth=2)

        # Store the information about the rectangle
        R["patch"] = new_rectangle
        R["path"] = new_rectangle.get_path().transformed(transform=new_rectangle.get_transform())
        R['label_offset'] = {"x": 0.0, "y": -10}

        # Information about the ranges
        R['xrange'] = [R['llx'], R['llx'] + R['width']]
        R['yrange'] = [R['lly'], R['lly'] + R['height']]

        R['xrange_pixel'] = np.floor([mc_layer.data_to_pixel(R['xrange'][0], 'x'),
                                      mc_layer.data_to_pixel(R['xrange'][1], 'x')])
        R['yrange_pixel'] = np.floor([mc_layer.data_to_pixel(R['yrange'][0], 'y'),
                                      mc_layer.data_to_pixel(R['yrange'][1], 'y')])

    for region in regions:
        # Next region
        R = regions[region]

        # The mapcube images a piece of Sun that inevitably moves.  Defining
        # the subcube using HPC co-ords will not work in this case.  Therefore
        # we need to convert the HPC co-ords into array indices and return
        # numpy arrays.  Create the subcube
        subdata = (mc.as_array())[R['yrange_pixel'][0]: R['yrange_pixel'][1],
                                  R['xrange_pixel'][0]: R['xrange_pixel'][1], :]

        # Region identifier name
        region_id = sd.ident + '_' + region

        # branch location
        b = [sd.corename, sd.sunlocation, sd.fits_level, sd.wave, region]

        # Output location
        output = sd.datalocationtools.save_location_calculator(sd.roots, b)["pickle"]

        # Output filename
        ofilename = os.path.join(output, region_id + '.datacube.pickle')

        # Open the file and write it out
        outputfile = open(ofilename, 'wb')
        pickle.dump(subdata, outputfile)
        pickle.dump(times, outputfile)
        outputfile.close()
        print('Saved to ' + ofilename)

#
# Save the regions
#

#
# Plot where the regions are
#
filepath = os.path.join(sd.save_locations['image'], sd.ident + '.regions.png')
step1_plots.plot_regions(mc_layer, regions, filepath)
