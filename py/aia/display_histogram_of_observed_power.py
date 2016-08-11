#
# Display a histogram of all the powers.
#
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

from tools import rnspectralmodels3
import analysis_get_data
import details_study as ds
import details_analysis as da
import details_plots as dp
import analysis_explore

# Paper 2: Wavelengths and regions we want to analyze
waves = ['94', '335', '131', '171', '193', '211']
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

# Models to fit
these_models = [rnspectralmodels3.PowerLawPlusConstantPlusLognormal(),
                rnspectralmodels3.PowerLawPlusConstant()]
n_models = len(these_models)

# Load the model fits
storage = analysis_get_data.get_all_data(waves=(waves[0],),
                                         regions=regions,
                                         spectral_model='.rnspectralmodels3')

# Define the masks
mdefine = analysis_explore.MaskDefine(storage, limits)
available_models = mdefine.available_models

# Which power to load
power_type = 'fourier_power'
# power_type = 'fourier_power_relative'

# Split each axis in to this number of sub-regions
nsr = 1

# Number of histogram bins
hist_bins = 200 // nsr

#
# Define some pixel locations
#
ny = storage[waves[0]][regions[0]][available_models[0]].ny
nx = storage[waves[0]][regions[0]][available_models[0]].nx
ylsr = ny // nsr
xlsr = nx // nsr

#
# Details of the plotting
#
linewidth = 3
log_x_axis = True
log_y_axis = False

if power_type == 'fourier_power_relative':
    ylim_power_type = [-4.0, 2.0]

if power_type == 'fourier_power':
    ylim_power_type = None


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

            # branch location
            b = [ds.corename, ds.sunlocation, ds.fits_level, wave, region]

            # Region identifier name
            region_id = ds.datalocationtools.ident_creator(b)

            # Different information criteria
            image = dp.get_image_model_location(ds.roots, b, [power_type, ])

            # General notification that we have a new data-set
            print('\nLoading New Data')
            # Which wavelength?
            print('Wave: ' + wave + ' (%i out of %i)' % (iwave + 1, len(waves)))
            # Which region
            print('Region: ' + region + ' (%i out of %i)' % (iregion + 1, len(regions)))
            # Which window
            print('Window: ' + window + ' (%i out of %i)' % (iwindow + 1, len(windows)))

            # Load the observed power spectra
            pkl_file_location = os.path.join(output, ofilename + '.%s.pkl' % power_type)
            print('Loading observed power spectra from ' + pkl_file_location)
            pkl_file = open(pkl_file_location, 'rb')
            pfrequencies = pickle.load(pkl_file)
            pwr = pickle.load(pkl_file)
            pkl_file.close()

            # Frequency in the units we need
            f = pfrequencies.to(dp.fz).value
            nf = len(f)
            hist_range = [np.nanmin(np.log10(pwr)), np.nanmax(np.log10(pwr))]
            pwr_hist = np.zeros((hist_bins, nf))
            log10_pwr_mean = np.zeros((nf,))
            log10_pwr_median = np.zeros_like(log10_pwr_mean)
            log10_pwr_mode = np.zeros_like(log10_pwr_mean)

            regular_subregions = []
            for i in range(0, nsr):
                for j in range(0, nsr):
                    regular_subregions.append([[j*ylsr, (j + 1)*ylsr], [i*xlsr, (i + 1)*xlsr]])

            for ir, regular_subregion in enumerate(regular_subregions):
                y_start = regular_subregion[0][0]
                y_end = regular_subregion[0][1]
                x_start = regular_subregion[1][0]
                x_end = regular_subregion[1][1]
                for this_f_index in range(0, nf):
                    log10_pwr = np.log10(pwr[y_start:y_end, x_start:x_end, this_f_index])
                    log10_pwr_mean[this_f_index] = np.nanmean(log10_pwr)
                    log10_pwr_median[this_f_index] = np.nanmedian(log10_pwr)
                    hist, bin_edges = np.histogram(log10_pwr,
                                                   bins=hist_bins,
                                                   range=hist_range)
                    log10_pwr_mode[this_f_index] = 0.5*(bin_edges[np.argmax(hist[:])] + bin_edges[np.argmax(hist[:]+1)] )
                    pwr_hist[:, this_f_index] = hist[:] / np.max(hist[:])
                spm = bin_edges[0:hist_bins]
                fig, ax = plt.subplots()
                if log_y_axis:
                    yformatter = plt.FuncFormatter(dp.log_10_product)
                    ax.set_yscale('log')
                    ax.yaxis.set_major_formatter(yformatter)
                if log_x_axis:
                    xformatter = plt.FuncFormatter(dp.log_10_product)
                    ax.set_xscale('log')
                    ax.xaxis.set_major_formatter(xformatter)
                #
                # Should also include a contour at powers_ratio = 1.0 to guide the eye
                # as to where the power ratio goes from above 1 (more power in the first
                # source) to less power (more power in the second source).
                #
                vmin = hist_range[0]
                vmax = hist_range[1]
                cax = ax.pcolormesh(f, spm, pwr_hist, cmap=cm.viridis)

                log10_pwr_mean_line = ax.plot(f, log10_pwr_mean, color='r', label='mean(log10(power))')
                log10_pwr_median_line = ax.plot(f, log10_pwr_median, color='r', linestyle='dashed', label='median(log10(power))')
                log10_pwr_mode_line = ax.plot(f, log10_pwr_mode, color='r', linestyle='dashdot', label='mode(log10(power))')

                power_label = r'log10(power) [range={:f}$\rightarrow${:f}]'.format(spm[0], spm[-1])
                frequency_label = r'frequency ({:s}) [range={:f}$\rightarrow${:f}]'.format(dp.fz, f[0], f[-1])
                ax.set_ylabel(power_label)
                ax.set_xlabel(frequency_label)
                ax.set_xlim(f[0], f[-1])
                if ylim_power_type is None:
                    ax.set_ylim(spm[0], spm[-1])
                else:
                    ax.set_ylim(ylim_power_type[0], ylim_power_type[1])
                ax.set_title(power_type)
                f5 = ax.axvline(dp.five_minutes.frequency.to(dp.fz).value,
                                linestyle=dp.five_minutes.linestyle,
                                color=dp.five_minutes.color, zorder=99,
                                label=dp.five_minutes.label)
                f3 = ax.axvline(dp.three_minutes.frequency.to(dp.fz).value,
                                linestyle=dp.three_minutes.linestyle,
                                color=dp.three_minutes.color, zorder=99,
                                label=dp.three_minutes.label)
                cb = fig.colorbar(cax, label='relative number')
                legend = ax.legend(loc='lower left', fontsize=10.0, framealpha=0.9)

                subregion = 'n=%s.N=%s.y0=%s.y1=%s.x0=%s.x1=%s' % (str(nsr), str(ir+1), str(y_start), str(y_end), str(x_start), str(x_end))
                final_filename = dp.concat_string([power_type, wave]) + '.%s.png' % str(subregion)
                # Dump the results as an image file
                final_filepath = os.path.join(image, final_filename)
                print('Saving to %s' % final_filepath)
                #plt.show()
                plt.savefig(final_filepath, bbox_inches='tight', pad_inches=0)
