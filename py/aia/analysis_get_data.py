#
# Loads the data and stores it in a sensible structure
#
#
# Analysis - distributions.  Load in all the data and make some population
# and spatial distributions
#
import os
import cPickle as pickle
import numpy as np
from matplotlib.collections import PolyCollection
import astropy.units as u

import study_details as sd
import sunpy.map
import sunpy.net.hek as hek
from sunpy.physics.transforms.solar_rotation import rot_hpc

#
# Function to get all the data in to one big dictionary
#
def get_all_data(waves=['171', '193'],
                 regions=['sunspot', 'moss', 'quiet Sun', 'loop footpoints'],
                 windows=['hanning'],
                 model_names=('Power law + Constant', 'Power law + Constant + Lognormal')):

    # Create the storage across all models, AIA channels and regions
    storage = {}
    for wave in waves:
        storage[wave] = {}
        for region in regions:
            storage[wave][region] = {}
            for model_name in model_names:
                storage[wave][region][model_name] = []

    #
    # Load in the fit results
    #
    for iwave, wave in enumerate(waves):
        for iregion, region in enumerate(regions):

            # branch location
            b = [sd.corename, sd.sunlocation, sd.fits_level, wave, region]

            # Region identifier name
            region_id = sd.datalocationtools.ident_creator(b)

            # Output location
            output = sd.datalocationtools.save_location_calculator(sd.roots, b)["pickle"]

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
#  Define a SunPy map
#
def make_map(output, region_id, map_data):
    # Get the map: Open the file that stores it and read it in
    map_data_filename = os.path.join(output, region_id + '.datacube.pkl')
    get_map_data = open(map_data_filename, 'rb')
    _dummy = pickle.load(get_map_data)
    _dummy = pickle.load(get_map_data)
    _dummy = pickle.load(get_map_data)
    _dummy = pickle.load(get_map_data)
    region_submap = pickle.load(get_map_data)
    # Specific region information
    get_map_data.close()

    # Create the map header
    header = {'cdelt1': region_submap.meta['cdelt1'],
              'cdelt2': region_submap.meta['cdelt2'],
              "crval1": region_submap.center[0].value,
              "crval2": region_submap.center[1].value,
              "telescop": region_submap.observatory,
              "detector": region_submap.observatory,
              "wavelnth": region_submap.measurement.value,
              "date-obs": region_submap.date}

    # Return a map using the input data
    return sunpy.map.Map(map_data, header)


def sunspot_outline():
    # -----------------------------------------------------------------------------
    # Get the sunspot details at the time of its detection
    #
    client = hek.HEKClient()
    qr = client.query(hek.attrs.Time("2012-09-23 01:00:00", "2012-09-23 02:00:00"), hek.attrs.EventType('SS'))
    p1 = qr[0]["hpc_boundcc"][9: -2]
    p2 = p1.split(',')
    p3 = [v.split(" ") for v in p2]
    p4 = np.asarray([(eval(v[0]), eval(v[1])) for v in p3])
    polygon = np.zeros([1, len(p2), 2])
    polygon[0, :, :] = p4[:, :]
    sunspot_date = qr[0]['event_endtime']
    return sunspot_date, polygon


def rotate_sunspot_outline(sunspot_date, polygon, date, linewidth=[5]):
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
                          edgecolors=['k'],
                          facecolors=['none'],
                          linewidth=linewidth)
