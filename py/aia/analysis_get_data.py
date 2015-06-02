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
import numpy as np
import analysis_details as ad
import sunpy.map


#
# Function to get all the data in to one big dictionary
#
def get_all_data(waves=['171', '193'],
                 regions=['sunspot', 'moss', 'quiet Sun', 'loop footpoints'],
                 windows=['hanning'],
                 model_names=('power law with constant and lognormal', 'power law with constant')):

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




def as_numpy_array(data, indices):
    ny = len(data)
    nx = len(data[0])
    results = np.zeros((ny, nx))
    for i in range(0, nx):
        for j in range(0, ny):
            x = data[j][i]
            for index in indices:
                x = x[index]
            results[j, i] = x
    return results


#
# Turn the input data in to a map
#
def make_map(output, region_id, wave, map_data):
    # Get the map: Open the file that stores it and read it in
    map_data_filename = os.path.join(output, region_id + '.datacube.pkl')
    get_map_data = open(map_data_filename, 'rb')
    _dummy = pickle.load(get_map_data)
    _dummy = pickle.load(get_map_data)
    # map layer
    mc_layer = pickle.load(get_map_data)
    # Specific region information
    get_map_data.close()

    # Create the map
    header = {'cdelt1': 0.6, 'cdelt2': 0.6, "crval1": -332.5, "crval2": 17.5,
          "telescop": 'AIA', "detector": "AIA", "wavelnth": wave,
          "date-obs": mc_layer.date}
    my_map = sunpy.map.Map(map_data, header)


#
# Define a dictionary that contains maps of the fit parameters of interest
#
def get_all_fit_quantities(pvalue, nposfreq, waves=['171', '193'],
                           regions=['sunspot', 'moss', 'quiet Sun', 'loop footpoints'],
                           windows=('hanning'),
                           model_names=('power law with constant and lognormal', 'power law with constant')):

    # Convert the parameters to more sensible values
    conversion_factor = ad.conversion(model_names)

    # Get the data
    storage = get_all_data(waves=waves, regions=regions,
                           model_names=model_names)

    # Define the rchi2 limits for each model name
    nparameters = ad.nparameters(model_names)
    rchi2limit = ad.rchi2limit(pvalue, nposfreq, nparameters)

    # Create the storage across all models, AIA channels and regions
    quantities = {}
    for wave in waves:
        quantities[wave] = {}
        for region in regions:
            quantities[wave][region] = {}
            for model_name in model_names:
                quantities[wave][region][model_name] = []

    for iwave, wave in enumerate(waves):
        for iregion, region in enumerate(regions):

            for imodel_name, model_name in enumerate(model_names):

                z = storage[wave][region][model_name]

                result = {"rchi2": as_numpy_array(z, ad.rchi2),
                          "aic": as_numpy_array(z, ad.ic["AIC"]),
                          "bic": as_numpy_array(z, ad.ic["BIC"]),
                          "success": as_numpy_array(z, ad.success)}

                # Good results mask
                parameters = ad.parameters(model_names)

                result["mask"] = ad.result_filter(result["success"], result["rchi2"], rchi2limit[model_name])

                # Put all the parameters in a separate dictionary entry
                for iparameter_name, parameter_name in enumerate(parameters[model_name]):
                    result[parameter_name] = as_numpy_array(z, [1, 'x', iparameter_name]) * conversion_factor[iparameter_name]

                # Final storage dictionary
                quantities[wave][region][model_name] = result
    return quantities
