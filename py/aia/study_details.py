#
# study details in one handy place.
#

import datalocationtools
import astropy.units as u

#
# Study type
#
#study_type = 'debugpaper2'
study_type = 'paper2'

# Target cadence
target_cadence = 12

# Target cadence
absolute_tolerance = 0.5 * u.s

# base cross-correlation channel
base_cross_correlation_channel = '171'

# Use base cross-correlation channel?
use_base_cross_correlation_channel = False

#
# Setup the details given the study type
#
if study_type == 'debugpaper2':
    dataroot = '~/Data/ts/'
    corename = 'request4'
    sunlocation = 'debug'
    fits_level = '1.5'
    wave = '171'
    cross_correlate = True
    derotate = True

if study_type == 'paper2':
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
    extension = 'cc_True_'
else:
    extension = 'cc_False_'

# Extend the name if derotation is requested
if derotate:
    extension = extension + 'dr_True_'
else:
    extension = extension + 'dr_False'

# Extend the name if derotation is requested
if use_base_cross_correlation_channel:
    extension = extension + 'bcc_True_'
else:
    extension = extension + 'bcc_False'



# Locations of the output datatypes
roots = {"pickle": '~/ts/pickle/' + extension,
         "image": '~/ts/img/' + extension,
         "movie": '~/ts/movies/' + extension}
save_locations = datalocationtools.save_location_calculator(roots, branches)

# Identity of the data
ident = datalocationtools.ident_creator(branches)