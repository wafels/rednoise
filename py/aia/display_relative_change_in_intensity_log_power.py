#
# Display a histogram of all the powers.
#
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from tools import rnspectralmodels3
import analysis_get_data
import details_study as ds
import details_analysis as da
import details_plots as dp
import analysis_explore

# Paper 2: Wavelengths and regions we want to analyze
waves = ['94', '131', '171', '193', '211', '335']
regions = ['six_euv']

# Wavelengths and regions we want to analyze
#waves = ['171', '193']
#regions = ['sunspot', 'quiet Sun']

# Paper 3: BM3D Wavelengths and regions we want to analyze
#waves = ['171']
#regions = ['six_euv']
#waves = ['171']
#regions = ['test_six_euv']

# Parameter limits
limit_type = 'standard'

#
# Parameter limits and information criteria
#
limits = da.limits[limit_type]
ic_types = da.ic_details.keys()

# Apodization windows
windows = ['hanning']

# Plotting
log_x_axis = True
log_y_axis = False


#
# Details of the plotting
#
linewidth = 3
plot_type = 'log_relative_power'

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
            pkl_file_location = os.path.join(output, ofilename + '.sum_log_fft_power_relative_intensities.npz')
            print('Loading average relative power spectra from ' + pkl_file_location)
            info = np.load(pkl_file_location)
            drel_power = info['drel_power']
            pfrequencies = 1000*info['pfrequencies']

            fig, ax = plt.subplots()
            ax.plot(pfrequencies, drel_power)
            ax.set_xscale('log')
            ax.set_title(wave)
            ax.set_xlabel('frequency (mHz)')
            ax.set_ylabel('average relative power')

            f5 = ax.axvline(dp.five_minutes.frequency.to(dp.fz).value,
                            linestyle=dp.five_minutes.linestyle,
                            color=dp.five_minutes.color, zorder=99,
                            label=dp.five_minutes.label)
            f3 = ax.axvline(dp.three_minutes.frequency.to(dp.fz).value,
                            linestyle=dp.three_minutes.linestyle,
                            color=dp.three_minutes.color, zorder=99,
                            label=dp.three_minutes.label)
            legend = ax.legend(loc='lower left', fontsize=10.0, framealpha=0.9)

            final_filename = dp.concat_string([plot_type, wave]) + '.png'
            # Dump the results as an image file
            final_filepath = os.path.join(image, final_filename)
            print('Saving to %s' % final_filepath)
            #plt.show()
            plt.savefig(final_filepath, bbox_inches='tight', pad_inches=0)
