#
# study details in one handy place.
#

import os
from copy import deepcopy
import numpy as np
import datalocationtools
import astropy.units as u
from sunpy.physics.differential_rotation import solar_rotate_coordinate
from sunpy.time import parse_time
from sunpy.map import Map

from matplotlib.patches import Polygon

#
# Study type
#

# Paper 2
#study_type = 'debugpaper2'
#study_type = 'paper2'
#study_type = 'vk'
#study_type = 'paper2_shorter_ts'

# 3 - Noise Analysis
#study_type = 'paper3_PSF'
#study_type = 'paper3_BM3D'
#study_type = 'paper3_BLSGSM'

# 4 - Simulated Data
#study_type = 'papern_bradshaw_simulation'
study_type = 'papern_bradshaw_simulation_low_fn'
#study_type = 'papern_bradshaw_simulation_intermediate_fn'
#study_type = 'papern_bradshaw_simulation_high_fn'
#study_type = 'from_simulated_power_spectra_1'
#study_type = 'from_simulated_power_spectra_10'

# 5 - GMU Study
#study_type = 'gmu1'

# 6 - Jitter
#study_type = 'jitter'

wave = '94'
#wave = '131'
#wave = '171'
#wave = '193'
#wave = '211'
#wave = '335'

input_root = '~/Data/ts'

# Target cadence
target_cadence = 12.0

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

# Standard index range of files to read in
file_list_index = [0, None]


sim_name = {'papern_bradshaw_simulation_low_fn': 'low occurrence rate nanoflares',
            'papern_bradshaw_simulation_intermediate_fn': 'intermediate occurrence rate nanoflares',
            'papern_bradshaw_simulation_high_fn': 'high occurrence rate nanoflares',
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
# Paper 2
#
if study_type == 'debugpaper2':
    dataroot = '~/Data/ts/'
    # corename = 'request4'
    corename = 'paper2_six_euv'
    sunlocation = 'debug'
    fits_level = '1.5'
    cross_correlate = True
    derotate = True

if study_type == 'paper2':
    dataroot = '~/Data/ts/'
    corename = 'paper2_six_euv'
    sunlocation = 'disk'
    fits_level = '1.5'
    cross_correlate = True
    derotate = True
    step0_output_information = ''
    step1_input_information = ''
    step1_output_information = ''
    rn_processing = ''
    regions = {"six_euv": {"llx": -500.0*u.arcsec, "lly": -100*u.arcsec,
                           "width": 340*u.arcsec, "height": 200*u.arcsec}}
    fevents = [('SS', 'EGSO_SFC')]

    study = Study(wave=wave,
                  input_root=input_root,
                  input_branches=['paper2_six_euv',
                                  'disk',
                                  '1.5',
                                  step0_output_information + step1_input_information +step1_output_information],
                  region=regions)


if study_type == 'paper2_shorter_ts':
    dataroot = '~/Data/ts/'
    corename = 'paper2_six_euv'
    sunlocation = 'disk'
    fits_level = '1.5'
    cross_correlate = True
    derotate = True
    step0_output_information = ''
    step1_input_information = ''
    step1_output_information = ''
    rn_processing = ''
    regions = {"six_euv": {"llx": -500.0*u.arcsec, "lly": -100*u.arcsec,
                           "width": 340*u.arcsec, "height": 200*u.arcsec}}
    file_list_index = [0, 1749]


###############################################################################
# Noise Analysis - paper 3
#
if study_type == 'paper3_BM3D':
    dataroot = '~/Data/ts/'
    corename = study_type
    sunlocation = 'disk'
    fits_level = '1.5'
    cross_correlate = True
    derotate = True
    step0_output_information = '.kirk=1-10'
    step1_input_information = '.kirk=1-10_ireland=step0'
    step1_output_information = '.1-11'

if study_type == 'paper3_PSF':
    dataroot = '~/Data/ts/'
    corename = study_type
    sunlocation = 'disk'
    fits_level = '1.5'
    cross_correlate = True
    derotate = True
    step0_output_information = '.kirk=1-4'
    step1_input_information = '.kirk=1-4_ireland=step0'
    step1_output_information = '.1-5'

if study_type == 'paper3_BLSGSM':
    dataroot = '~/Data/ts/'
    corename = study_type
    sunlocation = 'disk'
    fits_level = '1.5'
    cross_correlate = True
    derotate = True
    step0_output_information = '.kirk=1-10'
    step1_input_information = '.kirk=1-10_ireland=step0'
    step1_output_information = '.1-11'

###############################################################################
# Simulated Data - paper 4
#
if study_type in list(sim_name.keys()):
    conversion_style = 'simple'
    # Where the original data is, typically FITS or sav files
    dataroot = os.path.expanduser('~/Data/ts/')
    # Where the output data will be stored.
    output_root = os.path.expanduser('~/ts/output/')
    corename = study_type
    # A description of the data
    original_datatype = 'disk'

###############################################################################
# GMU - paper 5
#
if study_type == 'gmu1':
    dataroot = '~/Data/ts/'
    corename = study_type
    sunlocation = 'disk'
    fits_level = '1.5'
    cross_correlate = True
    derotate = True
    step0_output_information = ''
    step1_input_information = ''
    step1_output_information = ''
    rn_processing = ''
    regions = {"ch": {"llx": -90.0*u.arcsec, "lly": 120*u.arcsec,
                      "width": 405*u.arcsec, "height": 220*u.arcsec}}
    fevents = [('CH', 'SPoCA')]
###############################################################################
# Jitter - paper 6
#
if study_type == 'jitter':
    dataroot = '~/Data/ts/'
    corename = study_type
    sunlocation = 'overlaps_HiC'
    sunlocation = 'same_time_as_overlaps_HiC'
    fits_level = '1.0'
    cross_correlate = True
    derotate = True
    step0_output_information = ''
    step1_input_information = ''
    step1_output_information = ''
    rn_processing = ''

###############################################################################
# VK - paper 7
#
if study_type == 'vk':
    dataroot = '~/Data/ts/'
    corename = study_type
    sunlocation = 'vk2012'
    fits_level = '1.5'
    cross_correlate = True
    derotate = True
    step0_output_information = ''
    step1_input_information = ''
    step1_output_information = ''
    rn_processing = ''
    vk_x_range = (375, 924 + 1) * u.pix
    vk_y_range = (150, 649 + 1) * u.pix
    ar_x = (125, 350) * u.pix
    ar_y = (75, 300) * u.pix

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

# Index string
index_string = 't' + str(file_list_index[0]) + '_' + str(file_list_index[1])

# Create the branches in order
branches = [corename, wave, original_datatype]

# Create the AIA source data location
aia_data_location = datalocationtools.save_location_calculator({"aiadata": dataroot}, branches)

# Locations of the output datatypes
roots = {"project_data": os.path.join(output_root, 'project_data'),
         "image": os.path.join(output_root, 'image')}
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


def make_simple_map(wave, date, data):
    """
    Make a map using
    :param wave:
    :param date:
    :param data:
    :return:
    """
    header = {"CDELT1": 0.6, "CDELT2": 0.6, "WAVELNTH": np.float(wave),
              "INSTRUME": "AIA", "TELESCOP": "SDO", "WAVEUNIT": "Angstrom",
              "DATE-OBS": date}
    return Map(data, header)


class StudyBoundingBox:
    def __init__(self, ll, ur, time=None):
        """
        A simple bounding box object that holds spatial information and
        optionally, a time associated with the bounding box

        :param ll: lower left hand corner of the bounding box
        :param ur: upper right hand corner of the bounding box
        :param time: a time associated with the bounding box
        """
        self.ll = ll
        self.ur = ur
        self.width = ur[0] - ll[0]
        self.height = ur[1] - ur[1]
        self.time = time
        self.area = self.width * self.height

    # Check if this bounding box overlaps with another Bounding Box object
    # Adapted from http://stackoverflow.com/questions/27152904/calculate-overlapped-area-between-two-rectangles

    def _dx(self, b):
        axmax = self.ur[0].to(u.arcsec).value
        axmin = self.ll[0].to(u.arcsec).value
        bxmax = b.ur[0].to(u.arcsec).value
        bxmin = b.ll[0].to(u.arcsec).value
        return min(axmax, bxmax) - max(axmin, bxmin)

    def _dy(self, b):
        aymax = self.ur[1].to(u.arcsec).value
        aymin = self.ll[1].to(u.arcsec).value
        bymax = b.ur[1].to(u.arcsec).value
        bymin = b.ll[1].to(u.arcsec).value
        return min(aymax, bymax) - max(aymin, bymin)

    def overlap_exists(self, b):
        """
        Tests if the current BoundingBox overlap with another

        :param b: a BoundingBox object
        :return: True, if b overlaps spatially with the current BoundingBox.
        otherwise, False
        """
        if (self._dx(b) >= 0.0) and (self._dy(b) >= 0.0):
            return True
        else:
            return False

    def overlap_area(self, b):
        if self.overlap_exists(self, b):
            return self._dx(b) * self._dy(b)
        else:
            return None

    def solar_rotate(self, new_time):
        """
        Use solar rotation to move the BoundingBox
        :param new_time: the time the BoundingBox is moved to
        :return: the current BoundingBox is updated with the input time. The
        spatial position is updated according to solar rotation.
        """
        self.ll = solar_rotate_coordinate(self.ll, self.time, new_time)
        self.ur = solar_rotate_coordinate(self.ur, self.time, new_time)
        self.time = new_time
        return self


class StudyPolygon:
    def __init__(self, polygon, time=None, name=None, **polycollection_kwargs):
        """
        A simple closed polygon that holds spatial information and
        optionally, a time associated with the bounding box

        :param polygon :
        :param time: a time associated with the polygon
        :param name: a name associated with the polygon
        """
        self.polygon = polygon
        self.time = time
        self.name = name
        self.nvertex = self.polygon.shape[0]
        self.mpl_polygon = Polygon(self.polygon.to(u.arcsec).value, fill=False, color='k')

    def solar_rotate(self, new_time):
        """
        Use solar rotation to move the polygon
        :param new_time: the time the polygon is moved to
        :return: the current polygon is updated with the input time. The
        spatial position is updated according to solar rotation.
        """
        self.polygon = solar_rotate_coordinate(self.polygon, self.time, new_time)
        self.time = new_time
        return self


class StudyFevents:
    def __init__(self, fevents, _dt=30*u.day):
        """
        A list of fevents, with some convenient methods.
        """
        self.fevents = fevents
        self._dt = _dt.to(u.s).value

    def closest_in_time(self, date):
        dt = deepcopy(self._dt)
        for i, fevent in enumerate(self.fevents):
            this_dt = np.abs((parse_time(fevent.time) - parse_time(date)).total_seconds())
            if this_dt < dt:
                this_fevent = i
                dt = this_dt
        return self.fevents[this_fevent]

