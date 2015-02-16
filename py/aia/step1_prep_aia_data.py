"""
Load in the de-rotated and co-aligned data cube, and make some
cutouts.
"""

import os

import cPickle as pickle

from sunpy.time import parse_time
from sunpy.map import Map
import numpy as np
from matplotlib.patches import Rectangle

import sd
import step1_plots

# Load in the derotated data into a datacube
directory = sd.save_locations['pickle']
filename = sd.ident + '.full_mapcube.pkl'
print('Acquiring data from ' + sd.save_locations['pickle'])
print('Filename = ' + filename)
outputfile = open(os.path.join(directory, filename), 'rb')
data = pickle.load(outputfile)
layer_index = pickle.load(outputfile)
outputfile.close()


# Get the date and times from the original mapcube
date_obs = []
time_in_seconds = []
for m in data:
    date_obs.append(parse_time(m.date))
    time_in_seconds.append((date_obs[-1] - date_obs[0]).total_seconds())
times = {"date_obs": date_obs, "time_in_seconds": np.asarray(time_in_seconds)}

#
# Create some time-series for further analysis
#
if sd.sunlocation == 'disk':
    #
    # Define the number of regions and their extents.  Use Helio-projective
    # co-ordinates
    #
    regions = {"sunspot": {"llx": 0, "lly": 0, "width": 0, "height": 0},
               "loop footpoints": {"llx": 0, "lly": 0, "width": 0, "height": 0},
               "quiet Sun": {"llx": 0, "lly": 0, "width": 0, "height": 0},
               "active region": {"llx": 0, "lly": 0, "width": 0, "height": 0}}

    # Keep the rectangular patches
    keys = regions.keys()
    nregions = len(keys)
    patches = []
    for ir in range(0, nregions):
        # Next key
        key = keys[ir]
        region = regions[key]
        # Define a matplotlib rectangular patch to show the region on a map
        new_rectangle = Rectangle((region['llx'], region['lly']),
                                   region['width'], region['width'],
                                   label=key, fill=False,
                                   facecolor='b', edgecolor='r', linewidth=2)
        # Store the information about the region
        region["patch"] = new_rectangle,
        region["path"] = new_rectangle.get_path().transformed(transform=new_rectangle.get_transform())

    for region in regions:
        # Get the location of the regions we are interested in
        xlocation = [region['llx'], region['llx'] + region['width']]
        ylocation = [region['lly'], region['lly'] + region['height']]

        # Create the subcube
        submc = Map([m.submap(xlocation, ylocation) for m in data], cube=True)

        # Region identifier name
        region_id = ident + '_' + region

        # branch location
        b = [corename, sunlocation, fits_level, wave, region]

        # Output location
        output = datalocationtools.save_location_calculator(roots, b)["pickle"]

        # Output filename
        ofilename = os.path.join(output, region_id + '.mapcube.pickle')

        # Open the file and write it out
        outputfile = open(ofilename, 'wb')
        pickle.dump(submc, outputfile)
        pickle.dump(times, outputfile)
        outputfile.close()
        print('Saved to ' + ofilename)

#
# Plot where the regions are
#
step1_plots.plot_regions(data[layer_index], regions)
