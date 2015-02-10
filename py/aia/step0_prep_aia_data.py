"""
Load in the FITS files and write out a mapcube that has had the derotation
and co-alignment applied as necessary
"""

import os

import cPickle as pickle
from tools import datalocationtools
from tools import step1_plots

from sunpy.map import Map
from sunpy.image.coalignment import mapcube_coalign_by_match_template
from sunpy.physics.transforms.solar_rotation import mapcube_solar_derotate, calculate_solar_rotate_shift



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
    step1_plots.plot_shifts(sr_shifts,'shifts due to solar de-rotation',
                            layer_index, filepath=filepath)

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
    step1_plots.plot_shifts(cc_shifts, 'shifts due to cross correlation',
                            layer_index, filepath=filepath)

#
# Save the full dataset
#
directory = save_locations['pickle']
filename = ident + '.full_mapcube.pkl'
outputfile = open(os.path.join(directory, filename), 'wb')
pickle.dump(data, outputfile)
outputfile.close()
