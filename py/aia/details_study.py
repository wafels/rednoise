#
# study details in one handy place.
#

import os
import numpy as np

import astropy.units as u

import datalocationtools
from details_spectral_models import SelectModel

#
# Study type
#

# 2 - Bradshaw & Viall (20??) Simulated Data
#
# The prefix/suffix 'bv' will indicate the analysis of the
# simulated data from this paper.
#
study_type = 'bv_simulation_low_fn'
study_type = 'bv_simulation_intermediate_fn'
study_type = 'bv_simulation_high_fn'
wave = '94'
#wave = '131'
#wave = '171'
#wave = '193'
#wave = '211'
#wave = '335'


study_type = "verify_fitting"
wave = '000'
#wave = '050'
#wave = '100'
#wave = '150'
#wave = '200'
#wave = '250'
#wave = '300'
#wave = '350'
#wave = '400'

# Target cadence
absolute_tolerance = 0.5 * u.s

# base cross-correlation channel
base_cross_correlation_channel = '171'

# Use base cross-correlation channel?
use_base_cross_correlation_channel = False

# Start off by analyzing time series data
use_time_series_data = True

# Fixed AIA scale across all channels
arcsec_per_pixel_unit = u.arcsec / u.pix
fixed_aia_scale = {'x': 0.6*arcsec_per_pixel_unit,
                   'y': 0.6*arcsec_per_pixel_unit}

# Standard index range of files to read in
file_list_index = [0, None]


sim_name = {'bv_simulation_low_fn': 'low occurrence rate nanoflares',
            'bv_simulation_intermediate_fn': 'intermediate occurrence rate nanoflares',
            'bv_simulation_high_fn': 'high occurrence rate nanoflares',
            'from_simulated_power_spectra': 'simulated power spectra'}

for i in range(1, 11):
    s = str(i)
    key = 'from_simulated_power_spectra_{:s}'.format(s)
    value = 'simulated power spectra {:s}'.format(s)
    sim_name[key] = value

#
# Setup the details given the study type
#

###############################################################################
# Bradshaw & Viall (20??) simulated data.
#
if study_type in list(sim_name.keys()):
    conversion_style = 'simple'
    # Where the original data is, typically FITS or sav files
    dataroot = os.path.expanduser('~/Data/ts/')
    # Where the output data will be stored.
    output_root = os.path.expanduser('~/web/ts/output')
    corename = study_type
    # A description of the data
    original_datatype = 'disk'
    # A function that calculates the data filename
    def source_filename(study_type, w):
        if study_type == 'bv_simulation_low_fn':
            filename = 'low_fn_AIA_{:s}_noisy.fits'
        elif study_type == 'bv_simulation_intermediate_fn':
            filename = 'AIA_{:s}_noisy.fits'
        elif study_type == 'bv_simulation_high_fn':
            filename = 'high_fn_AIA_{:s}_noisy.fits'
        return filename.format(w)
    # All the wavelengths
    waves = ['94', '131', '171', '193', '211', '335']

###############################################################################
# Simulation of power spectra.
#
if 'verify_fitting' in study_type:
    #
    use_time_series_data = False


    # Where the original data is, typically FITS or sav files
    dataroot = os.path.expanduser('~/Data/ts/')

    # Where the output data will be stored.
    output_root = os.path.expanduser('~/web/ts/output')

    #
    corename = study_type

    # A description of the data
    original_datatype = 'disk'

    # All the wavelengths
    waves = [wave]

    # Bradshaw & Viall frequencies
    df = 0.000196078431372549 - 9.80392156862745e-05
    pfrequencies = 9.80392156862745e-05 + df * np.arange(424)
    nx = 256
    ny = 256
    alpha = int(wave)/100
    amplitude = 284220259296.4687  # 284220259296.4687 maximum over all BV data
    white_noise = 2.7307259040496703e-10  # 2.7307259040496703e-10 minimum over all BV data
    true_parameters = [amplitude, alpha, white_noise]
    simulation_model = SelectModel('pl_c').observation_model


# Index string
index_string = 't' + str(file_list_index[0]) + '_' + str(file_list_index[1])

# Create the branches in order
branches = [corename, original_datatype, wave]

# Create the AIA source data location
aia_data_location = datalocationtools.save_location_calculator({"aiadata": dataroot}, branches)

# Locations of the output datatypes
roots = {"project_data": os.path.join(output_root, 'project_data'),
         "image": os.path.join(output_root, 'image')}
save_locations = datalocationtools.save_location_calculator(roots, branches)


# Identity of the data
ident = datalocationtools.ident_creator(branches)
