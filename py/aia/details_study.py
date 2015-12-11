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

#study_type = 'paper3_PSF'
#study_type = 'paper3_BM3D'

# Target cadence
target_cadence = 12

# Target cadence
absolute_tolerance = 0.5 * u.s

# base cross-correlation channel
base_cross_correlation_channel = '171'

# Use base cross-correlation channel?
use_base_cross_correlation_channel = False

# Fixed AIA scale across all channels
arcsec_per_pixel_unit = u.arcsec / u.pix
fixed_aia_scale = {'x': 0.6*arcsec_per_pixel_unit,
                   'y': 0.6*arcsec_per_pixel_unit}

#
# Setup the details given the study type
#
if study_type == 'debugpaper2':
    dataroot = '~/Data/ts/'
    # corename = 'request4'
    corename = 'paper2_six_euv'
    sunlocation = 'debug'
    fits_level = '1.5'
    wave = '94'
    wave = '131'
    #wave = '171'
    #wave = '193'
    #wave = '211'
    #wave = '335'
    cross_correlate = True
    derotate = True

if study_type == 'paper2':
    dataroot = '~/Data/ts/'
    corename = 'paper2_six_euv'
    sunlocation = 'disk'
    fits_level = '1.5'
    wave = '94'
    wave = '131'
    wave = '171'
    wave = '193'
    wave = '211'
    wave = '335'
    cross_correlate = True
    derotate = True

if study_type == 'paper3_BM3D':
    dataroot = '~/Data/ts/'
    corename = study_type
    sunlocation = 'disk'
    fits_level = '1.5'
    wave = '171'
    cross_correlate = True
    derotate = True

if study_type == 'paper3_PSF':
    dataroot = '~/Data/ts/'
    corename = study_type
    sunlocation = 'disk'
    fits_level = '1.5'
    wave = '171'
    cross_correlate = True
    derotate = True

"""
if study_type == 'paper2':
    dataroot = '~/Data/ts/'
    corename = 'request4'
    sunlocation = 'disk'
    fits_level = '1.5'
    wave = '131'
    wave = '171'
    wave = '193'
    wave = '211'
    cross_correlate = True
    derotate = True
"""



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

# Limits
structure_location_limits = {'lo': 2.0 * u.mHz,
                             'hi': 7.0 * u.mHz}

# Cutdown map details
hsr2015_range_x = [-498.0, -498 + 340.0] * u.arcsec
hsr2015_range_y = [-98.0, 98.0] * u.arcsec
hsr2015_model_name = {'Power law + Constant + Lognormal': '2',
                      'Power law + Constant': '1'}

