"""
Load in the FITS files and write out a numpy arrays
"""

import os
import matplotlib.pyplot as plt

import cPickle as pickle
import aia_specific

from sunpy.time import parse_time
from sunpy.cm import cm
from sunpy.map import Map
from sunpy.image.coalignment import mapcube_coalign_by_match_template
from sunpy.image.solar_differential_rotation import mapcube_solar_derotate
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
from matplotlib import cm
from paper1 import sunday_name, label_sunday_name, figure_data_plot


# input data
dataroot = '~/Data/AIA/'
corename = 'study2'
sunlocation = 'equatorial'
fits_level = '1.5'
wave = '171'
cross_correlate = False
derotate = False

# Create the branches in order
branches = [corename, sunlocation, fits_level, wave]

# Create the AIA source data location
aia_data_location = aia_specific.save_location_calculator({"aiadata": dataroot}, branches)

# Extend the name if cross correlation is requested
extension = "_"
if cross_correlate:
    extension = extension + 'cc_True_'
else:
    extension = extension + 'cc_False_'

# Extend the name if derotation is requested
if derotate:
    extension = extension + 'dr_True_'
else:
    extension = extension + 'dr_False'

# Locations of the output datatypes
roots = {"pickle": '~/ts/pickle' + extension,
         "image": '~/ts/img' + extension,
         "movie": '~/ts/movies' + extension}
save_locations = aia_specific.save_location_calculator(roots, branches)

# Identity of the data
ident = aia_specific.ident_creator(branches)

# Load in the derotated data into a datacube
print('Loading' + aia_data_location["aiadata"])

# Get the list of data and sort it
list_of_data = sorted(os.path.join(aia_data_location["aiadata"], f) for f in os.listdir(aia_data_location["aiadata"]))
#
# Start manipulating the data
#
if derotate:
    data, derotation_displacements = mapcube_solar_derotate(Map(list_of_data, cube=True), with_displacements=True)
else:
    data = Map(list_of_data, cube=True)
    derotation_displacements = None

if cross_correlate:
    data, cross_correlation_displacements = mapcube_coalign_by_match_template(data, with_displacements=True)
else:
    cross_correlation_displacements = None

# Get the date and times from the original mapcube
date_obs = []
time_in_seconds = []
for m in data:
    date_obs.append(parse_time(m.date))
    time_in_seconds.append((date_obs[-1] - date_obs[0]).total_seconds())
times = {"date_obs": date_obs, "time_in_seconds": np.asarray(time_in_seconds)}

#
# Define the number of regions and their extents
#
nregions = 8
regions = {}
# X position
Xrange = data[0].xrange
start_xposition = 978.0
Xspacing = (Xrange[1] - start_xposition) / nregions
Xwidth = Xspacing - 5.0
# Y position
Yrange = data[1].yrange
Ymiddle = 0.0 #0.5 * (Yrange[0] + Yrange[1])
Ywidth = 50.0
Yregion = [Ymiddle - Ywidth, Ymiddle + Ywidth]
# Define the regions
# Keep the rectangular patches
patches = []
for i in range(0, nregions):
    # Next key
    key = "R" + str(i)
    # For Xwidth < Xspacing, no overlapping pixels
    Xregion = [start_xposition + i * Xspacing, start_xposition + i * Xspacing + Xwidth]
    regions[key] = {"x": Xregion, "y": Yregion}
    # Define a matplotlib rectangular patch to show the region on a map
    patches.append(Rectangle((Xregion[0], Yregion[0]), Xwidth, 2 * Ywidth, label=key, fill=False, facecolor='b', edgecolor='r', linewidth=2))

#
# Make a plot with the locations of the regions
#
fig, ax = plt.subplots()
data[0].plot()
for patch in patches:
    ax.add_patch(patch)
    llxy = patch.get_xy()
    plt.text(llxy[0] + 0.15 * Xwidth, llxy[1] - 15.0, patch.get_label(), bbox=dict(facecolor='w', alpha=0.5))
#plt.show()
plt.savefig(os.path.join(save_locations["image"], 'location.png'))

#
# Dump out the regions
#
for region in sorted(regions.keys()):
    # Get the region we are interested in
    xlocation = regions[region]['x']
    ylocation = regions[region]['y']
    submc = Map([m.submap(xlocation, ylocation) for m in data], cube=True)
    # Region identifier name
    region_id = ident + '_' + region
    # branch location
    b = [corename, sunlocation, fits_level, wave, region]
    # Output location
    output = aia_specific.save_location_calculator(roots, b)["pickle"]
    # Output filename
    ofilename = os.path.join(output, region_id + '.mapcube.pickle')
    # Open the file and write it out
    outputfile = open(ofilename, 'wb')
    pickle.dump(submc, outputfile)
    pickle.dump(times, outputfile)
    outputfile.close()
    print('Saved to ' + ofilename)

