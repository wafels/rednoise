#
# Loads the data and stores it in a sensible structure
#
#
# Analysis - distributions.  Load in all the data and make some population
# and spatial distributions
#
import os
from copy import deepcopy
import collections
import pickle
import numpy as np
import astropy.units as u

import details_study as ds
import sunpy.map
import sunpy.net.hek as hek


#
# Function to get all the data in to one big dictionary
#
def get_all_data(waves=['171', '193'],
                 regions=['sunspot', 'moss', 'quiet Sun', 'loop footpoints'],
                 windows=['hanning'],
                 model_names=('Power Law + Constant', 'Power Law + Constant + Lognormal'),
                 index_string='t0_None',
                 appended_name=None,
                 spectral_model=''):

    # Create the storage across all models, AIA channels and regions
    storage = {}
    for wave in waves:
        storage[wave] = {}
        for region in regions:
            storage[wave][region] = collections.OrderedDict()
            for model_name in model_names:
                storage[wave][region][model_name] = []

    #
    # Load in the fit results
    #
    for iwave, wave in enumerate(waves):
        for iregion, region in enumerate(regions):

            # branch location
            b = [ds.corename, ds.sunlocation, ds.fits_level, wave, region]

            # Region identifier name
            region_id = ds.datalocationtools.ident_creator(b)

            # Output location
            output = ds.datalocationtools.save_location_calculator(ds.roots, b)["pickle"]

            if appended_name is not None:
                output = output + appended_name

            # Go through all the windows
            for iwindow, window in enumerate(windows):

                # Output filename
                ofilename = os.path.join(output, region_id + '.datacube.{:s}.'.format(index_string) + window)

                # General notification that we have a new data-set
                print('Loading New Data')
                # Which wavelength?
                print('Wave: ' + wave + ' (%i out of %i)' % (iwave + 1, len(waves)))
                # Which region
                print('Region: ' + region + ' (%i out of %i)' % (iregion + 1, len(regions)))
                # Which window
                print('Window: ' + window + ' (%i out of %i)' % (iwindow + 1, len(windows)))

                # Load in the fit results
                filepath = os.path.join(output, ofilename + '%s.lnlike_fit_results.pkl' % spectral_model)
                print('Loading results to ' + filepath)
                f = open(filepath, 'rb')
                results = pickle.load(f)
                f.close()

                # Load in the emission results
                for itm, model_name in enumerate(model_names):
                    storage[wave][region][model_name] = results[model_name]
    return storage


#
# Get the region submap
#
def get_region_submap(output, region_id, average_submap=False, index_string='t0_None'):
    # Get the map: Open the file that stores it and read it in
    map_data_filename = os.path.join(output, region_id + '.datacube.{:s}.pkl'.format(index_string))
    print("Getting map data from %s " % map_data_filename)
    get_map_data = open(map_data_filename, 'rb')
    data = pickle.load(get_map_data)
    # Load everything else
    _dummy = pickle.load(get_map_data)
    _dummy = pickle.load(get_map_data)
    _dummy = pickle.load(get_map_data)
    region_submap = pickle.load(get_map_data)
    # Specific region information
    get_map_data.close()
    print(region_id + " region_submap: ",  region_submap.data.shape)

    # Return the maps using the input data
    meta = deepcopy(region_submap.meta)
    return {"reference region": region_submap,
            "mean": sunpy.map.Map(np.mean(data, axis=2), meta),
            "mean of log": sunpy.map.Map(np.mean(np.log(data), axis=2), meta),
            "standard deviation": sunpy.map.Map(np.std(data, axis=2), meta),
            "minimum": sunpy.map.Map(np.min(data, axis=2), meta),
            "maximum": sunpy.map.Map(np.max(data, axis=2), meta),
            "range": sunpy.map.Map(np.ptp(data, axis=2), meta),
            "median": sunpy.map.Map(np.median(data, axis=2), meta)}


#
#  Define a SunPy map
#
def make_map(region_submap, map_data):
    # Return a map using the input data
    return sunpy.map.Map(map_data, deepcopy(region_submap.meta))


#
# Cut down a map to some specific region
#
def hsr2015_map(m):
    submap = m.submap(ds.hsr2015_range_x, ds.hsr2015_range_y)
    return submap


def hsr2015_model_name(n):
    return ds.hsr2015_model_name[n]


#
# General feature/event download and processing routines
#
def _convert_hek_polygon(polygon_string):
    p1 = polygon_string[9: -2]
    p2 = p1.split(',')
    p3 = [v.split(" ") for v in p2]
    return np.asarray([(eval(v[0]), eval(v[1])) for v in p3])*u.arcsec


def _convert_hek_bbox_to_region(response):
    xmin = response['boundbox_c1ll']
    ymin = response['boundbox_c2ll']
    xmax = response['boundbox_c1ur']
    ymax = response['boundbox_c2ur']

    return ds.StudyBoundingBox((xmin, ymin)*u.arcsec, (xmax, ymax)*u.arcsec,
                          time=response['event_starttime'])


def fevent_outline(times, region_bbox, region_time, download=False,
                   fevent=('CH', 'SPoCA'), directory='~/',
                   filename='fevent.info.pkl'):
    """
    Get feature/event outlines from the HEK and save them in a format for use
    in further analyses.

    :param times : time range over which to search
    :param region_bbox : the region of the Sun each feature/event must overlap
    :param region_time :
    :param download: if True, then perform the query
    :param fevents: : list-like of the form [(event type1, [frm 11, frm12, ...]), (event type2, [frm 21, frm22, ...]), ...]
    :param directory: directory where the results are stored
    :param filename: filename of the stored results
    :return:
    """
    filepath = os.path.expanduser(os.path.join(directory, filename))
    if download:
        r = ds.StudyBoundingBox((region_bbox["llx"], region_bbox["lly"]),
                       (region_bbox["llx"] + region_bbox["width"], region_bbox["lly"] + region_bbox["height"]),
                       time=region_time)
        # Go through all the requested feature/event types and feature
        # recognition methods
        client = hek.HEKClient()
        fevent_type = fevent[0]
        fevent_frm = fevent[1]
        print("Acquiring {:s} frm={:s} data from the HEK".format(fevent_type, fevent_frm))
        qr = client.query(hek.attrs.Time(times[0], times[1]), hek.attrs.EventType(fevent_type))
        if len(qr) is None:
            shape_time = None
        else:
            shape_time = []
            for jjj, response in enumerate(qr):
                # If
                if response['frm_name'] == fevent_frm:
                    # Bounding box information for the fevent.  Needs to be
                    # rotated to the observation time
                    fevent_bbox = _convert_hek_bbox_to_region(response)
                    fevent_bbox = fevent_bbox.solar_rotate(region_time)
                    if r.overlap_exists(fevent_bbox):
                        polygon = _convert_hek_polygon(response['hpc_boundcc'])
                        shape_time.append(ds.StudyPolygon(polygon,
                                                     time=response['event_starttime'],
                                                     name=fevent_type))

        f = open(filepath, 'wb')
        pickle.dump(shape_time, f)
        f.close()
        print('Saved feature/event data to %s' % filepath)
    else:
        print("Acquiring feature/event data from %s" % filepath)
        f = open(filepath, 'rb')
        shape_time = pickle.load(f)
        f.close()
    return shape_time


#
# Code to estimate the warm component of AIA 94 and the Fe XVIII contribution.
#
def ugarte_warren_2014(data171, data193):
    """
    :param data171: AIA 171 data
    :param data193: AIA 193 data
    :return: an estimate of the warm component of the AIA 94 bandpass.

    Reference
    ---------
    Ugarte-Urra & Warren, ApJ, 783, 12, 2014.

    """
    # Parameter values for the fit (from the reference above).
    A = 0.39
    B = 116.32
    f = 0.31
    I_max = 27.5

    # The c array in this function needs to be reversed (compared to how it is
    # listed in the paper since we are choosing to use the np.polyval function
    # to evaluate the polynomial.
    c = np.asarray([-7.19/100.0, 9.75/1.0, 9.79/100.0, -2.81/1000.0])[::-1]

    # First order contribution (Equation 2 from the reference above).
    x = (f * data171 + (1.0 - f)*data193)/B
    x[x > I_max] = I_max

    # Calculate the warm component (using Equation 1 from the reference above).
    return A*np.polyval(c, x)


def estimated_aia94_warm_component(aia171, aia193, method=ugarte_warren_2014):
    """
    Wrapper around the Ugarte & Warren 2014 function that estimates the warm
    component of the AIA 94 data, returning a SunPy map from input SunPy
    AIA maps.

    :param aia171: an AIA 171 Angstrom map
    :param aia193: an AIA 193 Angstrom map
    :param method: the method used to estimate the AIA 94 warm component.
    :return: a SunPy map containing an estimate of warm component of the AIA 94
    bandpass. The output map has the same meta data as the input AIA 171
    map.

    References
    ----------
    Ugarte-Urra & Warren, ApJ, 783, 12, 2014.

    """
    if aia171.waveength.to(u.AA).value != 171:
        raise ValueError('First function argument must be an AIA 171 Angstrom SunPy map')
    elif aia193.waveength.to(u.AA).value != 193:
        raise ValueError('Second function argument must be an AIA 193 Angstrom SunPy map')

    # Calculate the warm data (using Equation 1 from the reference above) and
    # return a SunPy map.
    return sunpy.map.Map(method(aia171.data, aia193.data), aia171.meta)


def estimated_fe_xviii(aia94, aia171, aia193, method=ugarte_warren_2014):
    """
    :param aia94: an AIA 94 Angstrom map
    :param aia171: an AIA 171 Angstrom map
    :param aia193: an AIA 193 Angstrom map
    :param method: the method used to estimate the AIA 94 warm component.
    :return: a SunPy map containing an estimate of the Fe XVIII emission.  The
    output map has the same meta data as the input AIA 94 map.
    """
    if aia94.wavelength.to(u.AA).value != 94:
        raise ValueError('First function argument must be an AIA 94 Angstrom SunPy map')
    elif aia171.waveength.to(u.AA).value != 171:
        raise ValueError('Second function argument must be an AIA 171 Angstrom SunPy map')
    elif aia193.waveength.to(u.AA).value != 193:
        raise ValueError('Third function argument must be an AIA 193 Angstrom SunPy map')

    return sunpy.map.Map(aia94.data - method(aia171.data, aia193.data), aia94.meta)
