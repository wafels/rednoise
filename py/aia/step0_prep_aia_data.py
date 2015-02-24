"""
Step 0

Load in the FITS files and write out a mapcube that has had the derotation
and co-alignment applied as necessary.

For each channel, the solar derotation is calculated according to the time
stamps in the FITS file headers.

Image cross-correlation is applied using the shifts calculated by applying
sunpy's image co-alignment routine to the channel indicated by the variable
base_cross_correlation_channel.  This ensures that all channels get moved the
same way, and the shifts per channel do not depend on the structure or motions
in each channel.

"""

import os
import cPickle as pickle

import numpy as np

from sunpy.time import parse_time
from sunpy.map import Map
from sunpy.image.coalignment import mapcube_coalign_by_match_template
from sunpy.physics.transforms.solar_rotation import mapcube_solar_derotate, calculate_solar_rotate_shift

import step0_plots
import study_details as sd

# Base cross-correlation channel
base_cross_correlation_channel = '171'

# Create the AIA source data location
aia_data_location = sd.aia_data_location["aiadata"]

# Extend the name if cross correlation is requested
extension = sd.aia_data_location

# Locations of the output datatypes
save_locations = sd.save_locations

# Identity of the data
ident = sd.ident

# Load in the derotated data into a datacube
print('Acquiring data from ' + aia_data_location)

# Get the list of data and sort it
list_of_data = sorted(os.path.join(aia_data_location, f) for f in os.listdir(aia_data_location))
print("Number of files = %i" % len(list_of_data))
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
if sd.derotate:
    print("\nPerforming de-rotation")

    # Calculate the solar rotation of the mapcube
    print("Calculating the solar rotation shifts")
    sr_shifts = calculate_solar_rotate_shift(mc, layer_index=layer_index)

    # Plot out the solar rotation shifts
    filepath = os.path.join(save_locations['image'], ident + '.solar_derotation.png')
    step0_plots.plot_shifts(sr_shifts,'shifts due to solar de-rotation',
                            layer_index, filepath=filepath)
    filepath = os.path.join(save_locations['image'], ident + '.time.solar_derotation.png')
    step0_plots.plot_shifts(sr_shifts,'shifts due to solar de-rotation',
                            layer_index, filepath=filepath,
                            x=t_since_layer_index, xlabel='time relative to reference layer (s)')

    # Apply the solar rotation shifts
    data = mapcube_solar_derotate(mc, layer_index=layer_index, shifts=sr_shifts)
else:
    data = Map(list_of_data, cube=True)

#
# Coalign images by cross correlation
#
if sd.cross_correlate:
    ccbranches = [sd.corename, sd.sunlocation, sd.fits_level, base_cross_correlation_channel]
    ccsave_locations = sd.datalocationtools.save_location_calculator(sd.roots, ccbranches)
    ccident = sd.datalocationtools.ident_creator(ccbranches)
    ccfilepath = os.path.join(ccsave_locations['pickle'], ccident + '.cross_correlation.pkl')
    if sd.wave == base_cross_correlation_channel:
        print("\nPerforming cross_correlation and image shifting")
        data, cc_shifts = mapcube_coalign_by_match_template(data, layer_index=layer_index, with_displacements=True)
        print("Saving cross correlation shifts to %s" % filepath)
        f = open(ccfilepath, "wb")
        pickle.dump(cc_shifts, f)
        f.close()
    else:
        print("\nLoading in shifts to due cross-correlation from %s" % ccfilepath)
        f = open(ccfilepath, "rb")
        cc_shifts = pickle.load(f)
        f.close()
        print("Shifting images")
        data = mapcube_coalign_by_match_template(data, layer_index=layer_index, apply_displacements=cc_shifts)

    # Plot out the cross correlation shifts
    filepath = os.path.join(save_locations['image'], ident + '.cross_correlation.png')
    step0_plots.plot_shifts(cc_shifts, 'shifts due to cross correlation',
                            layer_index, filepath=filepath)
    filepath = os.path.join(save_locations['image'], ident + '.time.cross_correlation.png')
    step0_plots.plot_shifts(cc_shifts, 'shifts due to cross correlation',
                            layer_index, filepath=filepath,
                            x=t_since_layer_index, xlabel='time relative to reference layer (s)')
#
# Save the full dataset
#
directory = save_locations['pickle']
filename = ident + '.full_mapcube.pkl'
pfilepath = os.path.join(directory, filename)
print('Saving data to ' + pfilepath)
outputfile = open(pfilepath, 'wb')
pickle.dump(data, outputfile)
pickle.dump(layer_index, outputfile)
outputfile.close()
