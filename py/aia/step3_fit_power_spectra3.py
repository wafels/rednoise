#
# Step 3.  Load in the FFT power spectra and fit models.  Decide which one
# fits best.  Save the results.
#
# NOTE: USES VERSION 3 OF THE SPECTRAL MODELS
#
import pickle
import os

import details_study as ds
from tools import rnspectralmodels3

# Wavelengths and regions we want to analyze
#waves = ['171']
#regions = ['test_six_euv']

# Wavelengths and regions we want to analyze
# waves = ['131', '171', '193', '211', '335', '94']
waves = ['171', '193', '211', '335', '94', '131']
regions = ['six_euv']
power_type = 'fourier_power_relative'

# Fall AGU 2015 Wavelengths and regions we want to analyze
#waves = ['171', '193']
#regions = ['six_euv']


# Wavelengths and regions we want to analyze
#waves = ['171', '193']
#regions = ['sunspot', 'quiet Sun']

# BM3D Wavelengths and regions we want to analyze
#waves = ['171']
#regions = ['six_euv']

# Apodization windows
windows = ['hanning']

# Models to fit
these_models = [rnspectralmodels3.PowerLawPlusConstantPlusLognormal(),
                rnspectralmodels3.PowerLawPlusConstant()]
n_models = len(these_models)

#
# Main analysis loops
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
            ofilename = os.path.join(output, region_id + '.datacube.{:s}.'.format(ds.index_string) + window)

            # General notification that we have a new data-set
            print('\nLoading New Data')
            # Which wavelength?
            print('Wave: ' + wave + ' (%i out of %i)' % (iwave + 1, len(waves)))
            # Which region
            print('Region: ' + region + ' (%i out of %i)' % (iregion + 1, len(regions)))
            # Which window
            print('Window: ' + window + ' (%i out of %i)' % (iwindow + 1, len(windows)))

            # Load the data
            pkl_file_location = os.path.join(output, ofilename + '.{:s}.pkl'.format(power_type))
            print('Loading ' + pkl_file_location)
            pkl_file = open(pkl_file_location, 'rb')
            pfrequencies = pickle.load(pkl_file)
            pwr = pickle.load(pkl_file)
            pkl_file.close()

            # Storage for the results
            results = {}

            # Go through the models
            for itm, this_model in enumerate(these_models):
                print('Fitting model: %s (%i out of %i)' % (this_model.name, itm+1, n_models))

                # Do the fit and store the results for later analysis
                results[this_model.name] = rnspectralmodels3.Fit(pfrequencies.value, pwr, this_model, verbose=1)

            # Dump the results
            filepath = os.path.join(output, ofilename + '.rnspectralmodels3.lnlike_fit_results.pkl')
            print('Saving results to ' + filepath)
            f = open(filepath, 'wb')
            pickle.dump(results, f)
            f.close()
