#
# study details in one handy place.
#

import os
from copy import deepcopy
import numpy as np
import datalocationtools
import astropy.units as u
from sunpy.physics.differential_rotation import rot_hpc
from sunpy.time import parse_time

from matplotlib.patches import Polygon


class Study:
    """
    Defines all the major parameters used to study a single region in an
    input data set.
    """
    def __init__(self,
                 # Overall study name.
                 study_type='paper2',
                 # Which AIA channel to study
                 wave='171',
                 # Where the input data is
                 input_root=os.path.expanduser('~/Data/ts'),
                 input_branches=None,
                 # Output information
                 output_root=os.path.expanduser('~/ts'),
                 output_types=['pickle', 'images', 'movies'],
                 # Cross correlation and de-rotation parameters
                 cross_correlate=True,
                 derotate=True,
                 base_cross_correlation_channel='171',
                 use_base_cross_correlation=False,
                 # Spatial information
                 region=None,
                 # Temporal information
                 file_list_index=[0, None]):

        # Overall study name
        self.study_type = study_type

        # Which (AIA) channel to study
        self.wave = wave

        # Where the input data is
        self.input_root = input_root
        self.input_branches = input_branches

        # Output information
        self.output_root = output_root
        self.output_types = output_types

        # Information about procedures applied to the set of files in the
        # input directory.
        # Cross correlation and solar de-rotation parameters
        self.cross_correlate = cross_correlate
        self.base_cross_correlation_channel = base_cross_correlation_channel
        self.use_base_cross_correlation = use_base_cross_correlation
        self.derotate = derotate

        # Specific file indices that will be examined.
        self.file_list_index = file_list_index

        # Specific spatial subregions in the data
        if len(region) > 1:
            raise ValueError('Only one region allowed.')
        self.region = region

    def output_branches(self):
        """ Calculate the output branches. """
        # Overall study name.
        a = [self.study_type]
        # Copy the structure of the input data.
        for branch in self.input_branches:
            a.append(branch)
        # Copy the information about the procedures applied.
        for procedure in self.procedures_applied():
            a.append(procedure)
        # The wavelength used.
        a.append(self.wave)
        # Calculate an output branch for each region.
        a.append(list(self.region.keys())[0])
        return a

    def output_filepaths(self):
        """ Calculate the path for all the outputs."""
        a = dict()
        for output_type in self.output_types:
            root = datalocationtools.path([output_type], root=self.output_root)
            a[output_type] = datalocationtools.path(self.output_branches(),
                                                    root=root)
        return a

    def output_filename(self):
        """ Calculate a filename based on the input root and branches, and
        the procedures applied to that input data."""
        return datalocationtools.filename(self.output_branches())

    def info_file_list(self):
        # File list information as a string.
        return 't{:s}_{:s}'.format(str(self.file_list_index[0]),
                                   str(self.file_list_index[1]))

    def info_cc_and_dr(self):
        # Cross-correlation and solar de-rotation information as a string.
        return 'cc_{:s}_dr_{:s}_bcc_{:s}'.format(str(self.cross_correlate),
                                                 str(self.derotate),
                                                 str(self.use_base_cross_correlation))

    def procedures_applied(self):
        # Procedures applied to the data.
        a = list()
        a.append(self.info_file_list())
        a.append(self.info_cc_and_dr())
        return a



#
# Study type
#

# Paper 2
#study_type = 'debugpaper2'
study_type = 'paper2'
#study_type = 'paper2_shorter_ts'

# 3 - Noise Analysis
#study_type = 'paper3_PSF'
#study_type = 'paper3_BM3D'
#study_type = 'paper3_BLSGSM'

# 4 - Simulated Data
#study_type = 'papern_bradshaw_simulation'
#study_type = 'papern_bradshaw_simulation_low_fn'
#study_type = 'papern_bradshaw_simulation_intermediate_fn'
#study_type = 'papern_bradshaw_simulation_high_fn'
study_type = 'from_simulated_power_spectra_1'
study_type = 'from_simulated_power_spectra_6'

# 5 - GMU Study
#study_type = 'gmu1'

# 6 - Jitter
#study_type = 'jitter'

#wave = '94'
#wave = '131'
#wave = '171'
wave = '193'
#wave = '211'
wave = '335'

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


simulation = ['papern_bradshaw_simulation_low_fn',
              'papern_bradshaw_simulation_intermediate_fn',
              'papern_bradshaw_simulation_high_fn',
              'from_simulated_power_spectra']
for i in range(1, 11):
    simulation.append('from_simulated_power_spectra_{:s}'.format(str(i)))


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
if study_type in simulation:
    conversion_style = 'simple'
    dataroot = '~/Data/ts/'
    corename = study_type
    sunlocation = 'disk'
    fits_level = 'sim0'
    cross_correlate = True
    derotate = True
    step0_output_information = ''
    step1_input_information = ''
    step1_output_information = ''

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

# Processing information
processing_info = '{:s}.{:s}'.format(step1_input_information, index_string)

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
        self.ll = rot_hpc(self.ll[0], self.ll[1], self.time, new_time)
        self.ur = rot_hpc(self.ur[0], self.ur[1], self.time, new_time)
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
        rotated_polygon = np.zeros_like(self.polygon)
        for i in range(0, self.nvertex):
            rotated_vertex = rot_hpc(self.polygon[i, 0],
                                     self.polygon[i, 1], self.time, new_time)
            rotated_polygon[i, 0] = rotated_vertex[0]
            rotated_polygon[i, 1] = rotated_vertex[1]
        self.polygon = rotated_polygon
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
        for i, fevent in enumerate(fevents):
            this_dt = np.abs((parse_time(fevent.time) - date).total_seconds())
            if this_dt < dt:
                this_fevent = i
                dt = this_dt
        return self.fevents[this_fevent]

