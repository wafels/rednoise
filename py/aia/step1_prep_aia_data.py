"""
Load in the FITS files and write out a numpy arrays
"""

import os

import cPickle as pickle
from tools import datalocationtools
from tools import step1_plots

from sunpy.time import parse_time
from sunpy.map import Map
from sunpy.image.coalignment import mapcube_coalign_by_match_template
from sunpy.physics.transforms.solar_rotation import mapcube_solar_derotate, calculate_solar_rotate_shift
import numpy as np
from matplotlib.patches import Rectangle


# input data
dataroot = '~/Data/ts/'
corename = 'request4'
sunlocation = 'disk'
fits_level = '1.5'
wave = '171'
cross_correlate = True
derotate = True


# Create the branches in order
branches = [corename, sunlocation, fits_level, wave]

# Create the AIA source data location
aia_data_location = datalocationtools.save_location_calculator({"aiadata": dataroot}, branches)

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
save_locations = datalocationtools.save_location_calculator(roots, branches)

# Identity of the data
ident = datalocationtools.ident_creator(branches)

# Load in the derotated data into a datacube
print('Acquiring data from ' + aia_data_location["aiadata"])

# Get the list of data and sort it
list_of_data = sorted(os.path.join(aia_data_location["aiadata"], f) for f in os.listdir(aia_data_location["aiadata"]))
print("Number of files = %i" % len(list_of_data))
#
# Start manipulating the data
#
print("Loading data")
mc = Map(list_of_data, cube=True)

# Solar de-rotation and cross-correlation operations will be performed relative
# to the map at this index.
layer_index = len(mc) / 2

#
# Apply solar derotation
#
if derotate:
    print("Performing de-rotation")

    # Calculate the solar rotation of the mapcube
    print("Calculating the solar rotation shifts")
    sr_shifts = calculate_solar_rotate_shift(mc, layer_index=layer_index)

    # Plot out the solar rotation shifts
    filepath = os.path.join(save_locations['image'], ident + '.solar_derotation.png')
    step1_plots.plot_shifts(sr_shifts, 'solar de-rotation', layer_index,
                            filepath=filepath)

    # Apply the solar rotation shifts
    data = mapcube_solar_derotate(mc, layer_index=layer_index, shifts=sr_shifts)
else:
    data = Map(list_of_data, cube=True)

#
# Coalign images by cross correlation
#
if cross_correlate:
    print("Performing cross_correlation")
    data, cc_shifts = mapcube_coalign_by_match_template(data, layer_index=layer_index, with_displacements=True)

    # Plot out the cross correlation shifts
    filepath = os.path.join(save_locations['image'], ident + '.cross_correlation.png')
    step1_plots.plot_shifts(cc_shifts, 'cross correlation', layer_index,
                            filepath=filepath)

#
# Save the full dataset
#
directory = save_locations['pickle']
filename = ident + '.full_mapcube.pkl'
outputfile = open(os.path.join(directory, filename), 'wb')
pickle.dump(data, outputfile)
outputfile.close()

aaa = bbb

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
        output = datalocationtools.save_location_calculator(roots, b)["pickle"]
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
# Plot where the regions are
#
step1_plots.plot_regions(data[layer_index], regions)
