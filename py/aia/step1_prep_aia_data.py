"""
Load in the FITS files and write out a numpy arrays
"""

import os

import cPickle as pickle
import aia_specific

from sunpy.time import parse_time
from sunpy.map import Map
from sunpy.image.coalignment import mapcube_coalign_by_match_template
from sunpy.physics.transforms.solar_rotation import mapcube_solar_derotate
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt


# input data
dataroot = '~/Data/ts/'
corename = 'perpixel'
sunlocation = 'disk'
fits_level = '1.5'
wave = '193'
cross_correlate = True
derotate = True


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
# solar derotation
if derotate:
    data, derotation_displacements = mapcube_solar_derotate(Map(list_of_data, cube=True), with_displacements=True)
else:
    data = Map(list_of_data, cube=True)
    derotation_displacements = None

# Cross correlate
if cross_correlate:
    data, cross_correlation_displacements = mapcube_coalign_by_match_template(data, with_displacements=True)
else:
    cross_correlation_displacements = None

# rotate so that the limb appears to be at the equator
#if sunlocation in rotate_these:
#    rotate_angle = np.rad2deg(np.arctan(data[0].center['x'] / data[0].center['y']))
#    data = Map([m.rotate(-rotate_angle) for m in data], cube=True)


# Get the date and times from the original mapcube
date_obs = []
time_in_seconds = []
for m in data:
    date_obs.append(parse_time(m.date))
    time_in_seconds.append((date_obs[-1] - date_obs[0]).total_seconds())
times = {"date_obs": date_obs, "time_in_seconds": np.asarray(time_in_seconds)}

#
# Do the equatorial region
#
if sunlocation == 'disk':
    #
    # Define the number of regions and their extents.  Use Helio-projective
    # co-ordinates
    #
    region = {}

    nregions = 8
    regions = {}
    # X position
    Xrange = data[0].xrange
    start_xposition = 978.0
    Xspacing = (Xrange[1] - start_xposition) / nregions
    Rwidth = Xspacing - 5.0
    # Y position
    Yrange = data[1].yrange
    Ymiddle = 0.0 #0.5 * (Yrange[0] + Yrange[1])
    Ywidth = 50.0
    Yregion = [Ymiddle - Ywidth, Ymiddle + Ywidth]

    # Keep the rectangular patches
    patches = []
    for i in range(0, nregions):
        # Next key
        key = "R" + str(i)
        # For Xwidth < Xspacing, no overlapping pixels
        Xregion = [start_xposition + i * Xspacing, start_xposition + i * Xspacing + Rwidth]
        # Define a matplotlib rectangular patch to show the region on a map
        new_rectangle = Rectangle((Xregion[0], Yregion[0]), Rwidth, 2 * Ywidth, label=key, fill=False, facecolor='b', edgecolor='r', linewidth=2)
        # Store the information about the region
        regions[key] = {"x": Xregion, "y": Yregion,
                        "patch": new_rectangle,
                        "path": new_rectangle.get_path().transformed(transform=new_rectangle.get_transform()),
                        "radial_distance": start_xposition + i * Xspacing + 0.5 * Rwidth}

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

        ofilename = os.path.join(output, region_id + '.mapcube.radial_distance.pickle')
        # Open the file and write it out
        outputfile = open(ofilename, 'wb')
        pickle.dump(regions[region]["radial_distance"], outputfile)
        outputfile.close()
        print('Saved to ' + ofilename)



#
# Make a plot with the locations of the regions
#
fig, ax = plt.subplots()
z = data[0].plot()
#for patch in patches:
for region in sorted(regions.keys()):
    patch = regions[region]["patch"]
    ax.add_patch(patch)
    llxy = patch.get_xy()
    plt.text(llxy[0] + 0.15 * Rwidth, llxy[1] - 15.0, patch.get_label(), bbox=dict(facecolor='w', alpha=0.5))
#plt.show()
plt.savefig(os.path.join(save_locations["image"], 'location.png'))

