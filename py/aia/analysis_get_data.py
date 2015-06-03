#
# Loads the data and stores it in a sensible structure
#
#
# Analysis - distributions.  Load in all the data and make some population
# and spatial distributions
#
import os
import cPickle as pickle
import study_details as sd
import sunpy.map


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
    # map layer
    mc_layer = pickle.load(get_map_data)
    # Specific region information
    get_map_data.close()

    # Create the map header
    header = {'cdelt1': mc_layer.meta['cdelt1'],
              'cdelt2': mc_layer.meta['cdelt1'],
              "crval1": mc_layer.center[0].value,
              "crval2": mc_layer.center[1].value,
              "telescop": mc_layer.observatory,
              "detector": mc_layer.observatory,
              "wavelnth": mc_layer.measurement.value,
              "date-obs": mc_layer.date}

    # Return a map using the input data
    return sunpy.map.Map(map_data, header)
