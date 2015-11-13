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
import cPickle as pickle
import numpy as np
from matplotlib.collections import PolyCollection
import astropy.units as u

import details_study as ds
import sunpy.map
import sunpy.net.hek as hek
from sunpy.physics.transforms.solar_rotation import rot_hpc


#
# Function to get all the data in to one big dictionary
#
def get_all_data(waves=['171', '193'],
                 regions=['sunspot', 'moss', 'quiet Sun', 'loop footpoints'],
                 windows=['hanning'],
                 model_names=('Power Law + Constant', 'Power Law + Constant + Lognormal')):

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

            # Go through all the windows
            for iwindow, window in enumerate(windows):

                # Output filename
                ofilename = os.path.join(output, region_id + '.datacube.' + window)

                # General notification that we have a new data-set
                print('Loading New Data')
                # Which wavelength?
                print('Wave: ' + wave + ' (%i out of %i)' % (iwave + 1, len(waves)))
                # Which region
                print('Region: ' + region + ' (%i out of %i)' % (iregion + 1, len(regions)))
                # Which window
                print('Window: ' + window + ' (%i out of %i)' % (iwindow + 1, len(windows)))

                # Load in the fit results
                filepath = os.path.join(output, ofilename + '.lnlike_fit_results.pkl')
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
def get_region_submap(output, region_id, average_submap=False):
    # Get the map: Open the file that stores it and read it in
    map_data_filename = os.path.join(output, region_id + '.datacube.pkl')
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
    print region_id + " region_submap: ",  region_submap.data.shape

    # Return the maps using the input data
    meta = deepcopy(region_submap.meta)
    return {"reference region": region_submap,
            "mean": sunpy.map.Map(np.mean(data, axis=2), meta),
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


def rotate_sunspot_outline(polygon, sunspot_date, date, linewidth=[2], edgecolors=['k']):
    rotated_polygon = np.zeros_like(polygon)
    n = polygon.shape[1]
    for i in range(0, n):
        new_coords = rot_hpc(polygon[0, i, 0] * u.arcsec,
                             polygon[0, i, 1] * u.arcsec,
                             sunspot_date,
                             date)
        rotated_polygon[0, i, 0] = new_coords[0].value
        rotated_polygon[0, i, 1] = new_coords[1].value
    # Create the collection
    return PolyCollection(rotated_polygon,
                          alpha=1.0,
                          edgecolors=edgecolors,
                          facecolors=['none'],
                          linewidth=linewidth)

