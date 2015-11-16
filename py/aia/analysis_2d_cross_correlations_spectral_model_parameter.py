#
# Analysis - cross correlations
# Within a given model, cross correlate the model parameters.
# Two dimensional scatter plots and histograms are created from
# the good fits that are within the data; those fits must  also exceed the
# information criterion limit to be included.

import os
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import analysis_get_data
import analysis_explore
import details_study as ds
import details_plots as dp
import details_analysis as da

# Wavelengths we want to cross correlate
waves = ['131', '171', '193', '211', '94', '335']

# Regions we are interested in
regions = ['sunspot', 'moss', 'quiet Sun', 'loop footpoints']
regions = ['most_of_fov']
regions = ['four_wavebands']
regions = ['six_euv']

# Apodization windows
windows = ['hanning']

# Model results to examine
model_names = ('Power Law + Constant', 'Power Law + Constant + Lognormal')

#
# Details of the analysis
#
limits = da.limits
ic_types = da.ic_details.keys()

#
# Details of the plotting
#
fz = dp.fz
three_minutes = dp.three_minutes
five_minutes = dp.five_minutes
hloc = dp.hloc
linewidth = 3
bins = 100

# Load in all the data
storage = analysis_get_data.get_all_data(waves=waves,
                                         regions=regions,
                                         model_names=model_names)
mdefine = analysis_explore.MaskDefine(storage, limits)

# Plot cross-correlations within the same AIA channel
plot_type = 'cc.within'

# Different information criteria
for ic_type in ic_types:

    # Get the IC limit
    ic_limit = da.ic_details[ic_type]
    ic_limit_string = '%s>%f' % (ic_type, ic_limit)

    # Model name
    for this_model in model_names:

        # Select a region
        for region in regions:

            # Select a wave
            for wave in waves:

                # branch location
                b = [ds.corename, ds.sunlocation, ds.fits_level, wave, region]

                # Region identifier name
                region_id = ds.datalocationtools.ident_creator(b)

                # Output location
                output = ds.datalocationtools.save_location_calculator(ds.roots, b)["pickle"]

                # Get the parameter information
                this = storage[wave][region][this_model]

                # Parameters
                parameters = [v.fit_parameter for v in this.model.variables]
                npar = len(parameters)

                # Different information criteria
                image = dp.get_image_model_location(ds.roots, b, [this_model, ic_type])

                for i in range(0, npar):
                    # First parameter name
                    p1_name = parameters[i]

                    # First parameter, label for the plot
                    xlabel = this.model.variables[i].converted_label

                    # First parameter, data
                    p1 = this.as_array(p1_name)

                    # First parameter limits
                    p1_limits = limits[p1_name]

                    # Mask for the first and second parameters
                    mask1 = mdefine.combined_good_fit_parameter_limit[wave][region][this_model]
                    mask2 = mdefine.is_this_model_preferred(ic_type, ic_limit, this_model)[wave][region]
                    final_mask = np.logical_or(mask1, mask2)

                    # Get the final data for the first parameter
                    p1 = np.ma.array(this.as_array(p1_name), mask=final_mask).compressed()

                    # Unit
                    if str(this.model.variables[i].converted_unit) == '':
                        xunit = '(dimensionless)'
                    else:
                        xunit = '(%s)' % str(this.model.variables[i].converted_unit)

                    # Create the subtitle - model, region, information
                    # on how much of the field of view is not masked,
                    # and the information criterion and limit used.
                    ic_info_string = '%s, %s' % (ic_limit_string, dp.get_mask_info_string(final_mask))
                    subtitle = dp.concat_string([this_model,
                                                 '%s - %s' % (region, wave),
                                                 ic_info_string], sep='\n')
                    subtitle_filename = dp.concat_string([this_model,
                                                          region,
                                                          ic_info_string], sep='.')

                    for j in range(i+1, npar):
                        # Second parameter name
                        p2_name = parameters[j]

                        # Second parameter, label for the plot
                        ylabel = this.model.variables[j].converted_label

                        # Second parameter, data
                        p2 = this.as_array(p2_name)

                        # First parameter limits
                        p2_limits = limits[p2_name]

                        # Get the data using the final mask
                        p2 = np.ma.array(this.as_array(p2_name), mask=final_mask).compressed()

                        # Unit
                        if str(this.model.variables[j].converted_unit) == '':
                            yunit = '(dimensionless)'
                        else:
                            yunit = '(%s)' % str(this.model.variables[j].converted_unit)

                        # Cross correlation statistics
                        r = [spearmanr(p1, p2), pearsonr(p1, p2)]

                        # Form the rank correlation string
                        rstring = 'spr=%1.2f_pea=%1.2f' % (r[0][0], r[1][0])

                        # Plot identity as a file name
                        plot_identity_filename = dp.concat_string([rstring,
                                                                  wave,
                                                                  p1_name,
                                                                  p2_name])

                        # Make a scatter plot and the histogram
                        # Make the plot title
                        title = '%s vs. %s \n %s' % (xlabel, ylabel, subtitle)

                        for plot_function in (1, 2):
                            plt.close('all')
                            plt.title(title)
                            plt.xlabel('%s %s' % (xlabel, xunit))
                            plt.ylabel('%s %s' % (ylabel, yunit))
                            if plot_function == 1:
                                plt.scatter(p1, p2)
                                plot_name = 'scatter'
                            else:
                                plt.hist2d(p1, p2, bins=bins,
                                           range=[p1_limits.value, p2_limits.value])
                                plot_name = 'hist2d'
                            plt.xlim(p1_limits.value)
                            plt.ylim(p2_limits.value)
                            x0 = plt.xlim()[0]
                            ylim = plt.ylim()
                            y0 = ylim[0] + 0.3 * (ylim[1] - ylim[0])
                            y1 = ylim[0] + 0.6 * (ylim[1] - ylim[0])
                            plt.text(x0, y0, 'Pearson=%f' % r[0][0], bbox=dict(facecolor=dp.rchi2limitcolor[1], alpha=0.5))
                            plt.text(x0, y1, 'Spearman=%f' % r[1][0], bbox=dict(facecolor=dp.rchi2limitcolor[0], alpha=0.5))
                            final_filename = dp.concat_string([plot_type,
                                                               plot_identity_filename,
                                                               subtitle_filename,
                                                               plot_name]) + '.png'
                            plt.tight_layout()
                            plt.savefig(os.path.join(image, final_filename))
