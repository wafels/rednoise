"""
Step 1

Load in the de-rotated and co-aligned map cube, and make some cutouts that
focus on specific regions of interest.
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
    # co-ordinates.  These co-ordinates should be chosen in reference to the
    # time of the reference layer (layer_index).
    #
    regions = {"sunspot": {"llx": -335.0, "lly": 0, "width": 40, "height": 32},
               "loop footpoints": {"llx": -492, "lly": 0, "width": 23, "height": 22},
               "quiet Sun": {"llx": -200, "lly": -45, "width": 15, "height": 26},
               "moss": {"llx": -400, "lly": 25, "width": 45, "height": 25}}

    # Rectangular patches
    for region in regions:
        # Next region
        R = regions[region]

        # Define a matplotlib rectangular patch to show the region on a map
        new_rectangle = Rectangle((R['llx'], R['lly']),
                                  R['width'], R['height'],
                                  label=region, fill=False,
                                  facecolor='b', edgecolor='r', linewidth=2)

        # Store the information about the rectangle
        R["patch"] = new_rectangle
        R["path"] = new_rectangle.get_path().transformed(transform=new_rectangle.get_transform())
        R['label_offset'] = {"x": 0.0, "y": -10}

        # Information about the ranges
        R['xrange'] = [R['llx'], R['llx'] + R['width']]
        R['yrange'] = [R['lly'], R['lly'] + R['height']]

        R['llxy_pixel'] = np.floor([mc_layer.data_to_pixel(R['llx'], 'x'),
                                    mc_layer.data_to_pixel(R['lly'], 'y')])

        # Small changes in the plate scale in each AIA channel can mean that
        # the height and widths of each region depend on the data. To mitigate
        # against this we fix the size of the plate scale.
        fixed_aia_scale = {'x': 0.6, 'y': 0.6}
        R['width_pixel'] = np.floor(R['width'] / fixed_aia_scale['x'])
        R['height_pixel'] = np.floor(R['height'] / fixed_aia_scale['y'])

    for region in regions:
        # Next region
        R = regions[region]

        # The mapcube images a piece of Sun that inevitably moves.  Defining
        # the subcube using HPC co-ords will not work in this case.  Therefore
        # we need to convert the HPC co-ords into array indices and return
        # numpy arrays.  Create the subcube
        subdata = (mc.as_array())[R['llxy_pixel'][1]: R['llxy_pixel'][1] + R['height_pixel'],
                                  R['llxy_pixel'][0]: R['llxy_pixel'][0] + R['width_pixel'], :]

        print('\nSub datacube (size %i, %i, %i)' % subdata.shape)

        # Region identifier name
        region_id = sd.ident + '_' + region

        # branch location
        b = [sd.corename, sd.sunlocation, sd.fits_level, sd.wave, region]

        # Output location
        output = sd.datalocationtools.save_location_calculator(sd.roots, b)["pickle"]

        # Output filename
        ofilename = os.path.join(output, region_id + '.datacube.pkl')

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

        # Close the data
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
