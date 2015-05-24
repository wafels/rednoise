#
# Step 3.  Load in the FFT power spectra and fit models.  Decide which one
# fits best.  Save the results.
#
import cPickle as pickle
import os

import study_details as sd
import rnspectralmodels
import lnlike_model_fit
import pstools

# Wavelengths we want to analyze
waves = ['171', '193']

# Regions we are interested in
regions = ['sunspot', 'loop footpoints', 'quiet Sun', 'moss']

# Apodization windows
windows = ['hanning']

# Models to fit
these_models = {'power law with constant and lognormal': rnspectralmodels.power_law_with_constant_with_lognormal,
                'power law with constant': rnspectralmodels.power_law_with_constant}
nmodels = len(these_models)

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

            # Normalize the frequencies
            normed_freqs = pfrequencies / pfrequencies[0]

            # Size of the data
            nx = pwr.shape[1]
            ny = pwr.shape[0]

            # Storage for the results
            results = {}

            # Go through the models
            for itm, model_name in enumerate(these_models):
                # Next model
                model_function = these_models[model_name]
                print('Fitting model: %s (%i out of %i)' % (model_name, itm + 1, nmodels))

                # Storage for the results
                results[model_name] = [[None]*nx for i in range(ny)]

                for i in range(0, nx):
                    for j in range(0, ny):
                        # Get the next power spectrum
                        this = pwr[j, i, :]
                        # Generate an initial guess
                        initial_guess = pstools.generate_initial_guess(model_name, normed_freqs.value, this)

                        # Number of data points to fit
                        n = len(normed_freqs)

                        # Number of parameters
                        k = len(initial_guess)

                        # Do the fit
                        result = lnlike_model_fit.go(normed_freqs, this,
                                                     model_function, initial_guess,
                                                     "Nelder-Mead")

                        # Calculate Nita et al reduced chi-squared value
                        bestfit = model_function(result['x'], normed_freqs)

                        # Sample to model ratio
                        rhoj = lnlike_model_fit.rhoj(this, bestfit)

                        # Nita et al (2014) value of the reduced chi-squared
                        rchi2 = lnlike_model_fit.rchi2(1, n - k - 1, rhoj)

                        # AIC value
                        aic = lnlike_model_fit.AIC(k,
                                                   result['x'],
                                                   normed_freqs.value,
                                                   this, model_function)

                        # BIC value
                        bic = lnlike_model_fit.BIC(len(initial_guess),
                                                   result['x'],
                                                   normed_freqs.value,
                                                   this, model_function,
                                                   n)

                        # Store the results
                        results[model_name][j][i] = (initial_guess, result,
                                                     rchi2, aic, bic)

            # Dump the results
            filepath = os.path.join(output, ofilename + '.lnlike_fit_results.pkl')
            print('Saving results to ' + filepath)
            f = open(filepath, 'wb')
            pickle.dump(results, f)
            f.close()
