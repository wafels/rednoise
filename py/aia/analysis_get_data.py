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
from matplotlib.collections import PolyCollection
from matplotlib.patches import Polygon
import astropy.units as u

import details_study as ds
import sunpy.map
import sunpy.net.hek as hek
from sunpy.physics.solar_rotation import rot_hpc


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


def sunspot_outline(directory='~/ts/pickle/', filename='sunspot.info.pkl'):
    # -----------------------------------------------------------------------------
    # Get the sunspot details at the time of its detection
    #
    filepath = os.path.expanduser(os.path.join(directory, filename))
    if os.path.isfile(filepath):
        print("Acquiring sunspot data from %s" % filepath)
        f = open(filepath, 'rb')
        polygon = pickle.load(f)
        sunspot_date = pickle.load(f)
        f.close()
    else:
        print("Acquiring sunspot data from the HEK...")
        client = hek.HEKClient()
        qr = client.query(hek.attrs.Time("2012-09-23 01:00:00", "2012-09-23 02:00:00"), hek.attrs.EventType('SS'))
        p1 = qr[0]["hpc_boundcc"][9: -2]
        p2 = p1.split(',')
        p3 = [v.split(" ") for v in p2]
        p4 = np.asarray([(eval(v[0]), eval(v[1])) for v in p3])
        polygon = np.zeros([1, len(p2), 2])
        polygon[0, :, :] = p4[:, :]
        sunspot_date = qr[0]['event_starttime']

        f = open(filepath, 'wb')
        pickle.dump(polygon, f)
        pickle.dump(sunspot_date, f)
        f.close()
        print('Saved sunspot data to %s' % filepath)
    return polygon, sunspot_date


def fe_outline(times, data_box=None, download=True, fevents=['SS'],
               directory='~/', filename='fevent.info.pkl'):
    """
    :param times
    :param download:
    :param fevents:
    :param directory:
    :param filename:
    :return:
    """
    filepath = os.path.expanduser(os.path.join(directory, filename))
    if download:
        for fevent in fevents:
            print("Acquiring {:s} data from the HEK".format(fevent))
            client = hek.HEKClient()
            qr = client.query(hek.attrs.Time(times[0], times[1]), hek.attrs.EventType(fevent))
            if len(qr) is None:
                shape_time = None
            else:
                shape_time = []
                for response in qr:
                    # Check to see if the bounding box of the fevent overlaps
                    # with the bounding box extent of the data
                    bx0 = data_box.bottomleft.x
                    bx1 = data_box.topright.x
                    by0 = data_box.bottomleft.y
                    by1 = data_box.topright.y

                    fx0 = data_box.bottomleft.x
                    fx1 = data_box.topright.x
                    fy0 = data_box.bottomleft.y
                    fy1 = data_box.topright.y

                    x_extent = np.max([bx1, fx1]) - np.min([bx0, fx0])
                    y_extent = np.max([by1, fy1]) - np.min([by0, fy0])
                    if (x_extent < bx1-bx0 + fx1-fx0) and (y_extent < by1-by0 + fy1-fy0):
                        p1 = response["hpc_boundcc"][9: -2]
                        p2 = p1.split(',')
                        p3 = [v.split(" ") for v in p2]
                        p4 = np.asarray([(eval(v[0]), eval(v[1])) for v in p3])
                        polygon = np.zeros([1, len(p2), 2])
                        polygon[0, :, :] = p4[:, :]
                        fe_date = response['event_starttime']
                        shape_time.append((fevent, polygon, fe_date))

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


def rotate_fevent_outline(polygon, fevent_date, date, linewidth=[2], edgecolors=['k']):
    rotated_polygon = np.zeros_like(polygon)
    n = polygon.shape[1]
    for i in range(0, n):
        new_coords = rot_hpc(polygon[0, i, 0] * u.arcsec,
                             polygon[0, i, 1] * u.arcsec,
                             fevent_date,
                             date)
        rotated_polygon[0, i, 0] = new_coords[0].value
        rotated_polygon[0, i, 1] = new_coords[1].value

    # Create a matplotlib polygon
    mpl_polygon = Polygon(rotated_polygon[0, :, :])

    # Return the matplotlib polygon and the PolyCollection
    return mpl_polygon, PolyCollection(rotated_polygon,
                                       alpha=1.0,
                                       edgecolors=edgecolors,
                                       facecolors=['none'],
                                       linewidth=linewidth)


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
