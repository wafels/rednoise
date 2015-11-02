#
# Step 3.  Load in the FFT power spectra and fit models.  Decide which one
# fits best.  Save the results.
#
import cPickle as pickle
import os

import study_details as sd
import rnspectralmodels2

# Wavelengths we want to analyze
waves = ['131', '171', '193', '211']

# Regions we are interested in
#regions = ['sunspot', 'quiet Sun']
#regions = ['most_of_fov']
regions = ['four_wavebands']

# Apodization windows
windows = ['hanning']

# Models to fit
these_models = [rnspectralmodels2.PowerLawPlusConstantPlusLognormal(),
                rnspectralmodels2.PowerLawPlusConstant()]
n_models = len(these_models)

#
# Main analysis loops
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
            print('\nLoading New Data')
            # Which wavelength?
            print('Wave: ' + wave + ' (%i out of %i)' % (iwave + 1, len(waves)))
            # Which region
            print('Region: ' + region + ' (%i out of %i)' % (iregion + 1, len(regions)))
            # Which window
            print('Window: ' + window + ' (%i out of %i)' % (iwindow + 1, len(windows)))

            # Load the data
            pkl_file_location = os.path.join(output, ofilename + '.fourier_power.pkl')
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
                results[this_model.name] = rnspectralmodels2.Fit(pfrequencies.value, pwr, this_model)

            # Dump the results
            filepath = os.path.join(output, ofilename + '.lnlike_fit_results.pkl')
            print('Saving results to ' + filepath)
            f = open(filepath, 'wb')
            pickle.dump(results, f)
            f.close()
