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
models = (rnspectralmodels.power_law,
          rnspectralmodels.power_law_with_constant,
          rnspectralmodels.power_law_with_constant_with_lognormal)
model_names = ('power law with constant and lognormal',
               'power law with constant')

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
            for itm, this_model in enumerate(models):
                # Next model
                model_name = model_names[itm]
                print('Fitting model: %s (%i out of %i)' %(model_name, itm + 1, len(model_names)))

                # Storage for the results
                results[model_name] = [[None]*nx for i in range(ny)]

                for i in range(0, nx):
                    for j in range(0, ny):
                        # Get the next power spectrum
                        this = pwr[j, i, :]
                        # Generate an initial guess
                        initial_guess = pstools.generate_initial_guess(model_name, normed_freqs, this)

                        # Do the fit
                        result = lnlike_model_fit.go(normed_freqs, this,
                                                     this_model, initial_guess,
                                                     "Nelder-Mead")

                        # Calculate Nita et al reduced chi-squared value
                        bestfit = this_model(result['x'], normed_freqs)

                        # Sample to model ratio
                        rhoj = lnlike_model_fit.rhoj(this, bestfit)

                        # Nita et al (2014) value of the reduced chi-squared
                        rchi2 = lnlike_model_fit.rchi2(1, len(normed_freqs) - len(initial_guess) - 1, rhoj)

                        # Store the results
                        results[model_name][j][i] = (initial_guess, result, rchi2)

            # Dump the results
            filepath = os.path.join(output, ofilename + '.lnlike_fit_results.pkl')
            print('Saving results to ' + filepath)
            f = open(filepath, 'wb')
            pickle.dump(results, f)
            f.close()
