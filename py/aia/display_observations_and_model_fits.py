#
# Step 3.  Load in the FFT power spectra and fit models.  Decide which one
# fits best.  Save the results.
#
import cPickle as pickle
import os
import numpy as np
import matplotlib.pyplot as plt

import rnspectralmodels2
import analysis_get_data
import details_study as ds
import details_analysis as da
import details_plots as dp
import analysis_explore

# Paper 2: Wavelengths and regions we want to analyze
waves = ['131', '171', '193', '211', '335', '94']
regions = ['six_euv']

# Wavelengths and regions we want to analyze
#waves = ['171', '193']
#regions = ['sunspot', 'quiet Sun']

# Paper 3: BM3D Wavelengths and regions we want to analyze
waves = ['171']
regions = ['six_euv']

# Number of locations to print out
n_locations = 10

# Parameter limits
limit_type = 'standard'

#
# Parameter limits and information criteria
#
limits = da.limits[limit_type]
ic_types = da.ic_details.keys()

# Apodization windows
windows = ['hanning']

# Models to fit
these_models = [rnspectralmodels2.PowerLawPlusConstantPlusLognormal(),
                rnspectralmodels2.PowerLawPlusConstant()]
n_models = len(these_models)

# Load the model fits
storage = analysis_get_data.get_all_data(waves=waves,
                                         regions=regions)

# Define the masks
mdefine = analysis_explore.MaskDefine(storage, limits)
available_models = mdefine.available_models


# Define some pixel locations
ny = storage[waves[0]][regions[0]][available_models[0]].ny
nx = storage[waves[0]][regions[0]][available_models[0]].nx
these_locations = zip(np.random.randint(0, ny, size=n_locations),
                      np.random.randint(0, nx, size=n_locations))

these_locations = ((12, 325), (31,29), (67,335), (85,565), (92, 499), (101, 98), (110, 287), (121, 35), (132, 461), (192, 388))

#
# Details of the plotting
#
fz = dp.fz
three_minutes = dp.three_minutes
five_minutes = dp.five_minutes
hloc = dp.hloc
linewidth = 3
plot_type = 'observations_vs_model_fits'

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
            ofilename = os.path.join(output, region_id + '.datacube.' + window)

            # branch location
            b = [ds.corename, ds.sunlocation, ds.fits_level, wave, region]

            # Region identifier name
            region_id = ds.datalocationtools.ident_creator(b)

            # Different information criteria
            image = dp.get_image_model_location(ds.roots, b, [plot_type, ])

            # General notification that we have a new data-set
            print('\nLoading New Data')
            # Which wavelength?
            print('Wave: ' + wave + ' (%i out of %i)' % (iwave + 1, len(waves)))
            # Which region
            print('Region: ' + region + ' (%i out of %i)' % (iregion + 1, len(regions)))
            # Which window
            print('Window: ' + window + ' (%i out of %i)' % (iwindow + 1, len(windows)))

            # Load the observed power spectra
            pkl_file_location = os.path.join(output, ofilename + '.fourier_power.pkl')
            print('Loading observed power spectra from ' + pkl_file_location)
            pkl_file = open(pkl_file_location, 'rb')
            pfrequencies = pickle.load(pkl_file)
            pwr = pickle.load(pkl_file)
            pkl_file.close()

            # Frequency in the units we need
            f = pfrequencies.to(fz).value

            best_fits = {}
            bics = {}
            for model in available_models:
                best_fits[model] = storage[wave][region][model].best_fit()
                bics[model] = storage[wave][region][model].as_array('BIC')

            # Go through the pixel locations
            for this_y, this_x in these_locations:
                this_xy_string = 'y=%s,x=%s' % (this_y, this_x)
                title = dp.concat_string([wave, region, this_xy_string], sep='.')

                # Set up the plot
                plt.close('all')
                plt.loglog(f, pwr[this_y, this_x, :], label='observed power spectrum')

                # Go through each model and plot its components, and its sum
                for model in available_models:
                    plt.loglog(f, best_fits[model][this_y, this_x],
                               label='%s, BIC=%f' % (model, bics[model][this_y, this_x]))

                plt.axvline((1.0/five_minutes.position).to(fz).value,
                            color=five_minutes.color,
                            label=five_minutes.label,
                            linestyle=five_minutes.linestyle,
                            linewidth=five_minutes.linewidth)

                plt.axvline((1.0/three_minutes.position).to(fz).value,
                            color=three_minutes.color,
                            label=three_minutes.label,
                            linestyle=three_minutes.linestyle,
                            linewidth=three_minutes.linewidth)
                plt.ylabel('frequency (%s)' % fz)
                plt.xlabel('power (arb. units)')
                plt.title(title)
                plt.legend(fontsize=9, framealpha=0.5)
                plt.tight_layout()

                final_filename = dp.concat_string([plot_type, title]) + '.png'
                # Dump the results as an image file
                final_filepath = os.path.join(image, final_filename)
                print 'Saving to %s' % final_filepath
                plt.savefig(final_filepath)