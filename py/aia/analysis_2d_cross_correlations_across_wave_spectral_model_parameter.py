#
# Analysis - cross correlations
# Using the same model and region, cross correlate parameters across AIA
# channels. Two dimensional scatter plots and histograms are created from
# the good fits that are within the data; those fits must  also exceed the
# information criterion limit to be included.
#
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
waves = ['171', '193']

# Regions we are interested in
regions = ['sunspot', 'moss', 'quiet Sun', 'loop footpoints']
regions = ['most_of_fov']

# Apodization windows
windows = ['hanning']

# Model results to examine
model_names = ('Power law + Constant + Lognormal', 'Power law + Constant')

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


# Load in all the data
storage = analysis_get_data.get_all_data(waves=waves,
                                         regions=regions,
                                         model_names=model_names)
mdefine = analysis_explore.MaskDefine(storage, limits)

# Plot cross-correlations across different AIA channels
plot_type = 'cc.across'

# Different information criteria
for ic_type in ic_types:

    # Get the IC limit
    ic_limit = da.ic_details[ic_type]
    ic_limit_string = '%s>%f' % (ic_type, ic_limit[ic_type])

    # Model name
    for this_model in model_names:

        # Select a region
        for region in regions:

            # First wave
            for iwave1, wave1 in enumerate(waves):

                # branch location
                b = [ds.corename, ds.sunlocation, ds.fits_level, wave1, region]

                # Region identifier name
                region_id = ds.datalocationtools.ident_creator(b)

                # Output location
                output = ds.datalocationtools.save_location_calculator(ds.roots, b)["pickle"]

                # Image output location
                image = dp.get_image_model_location(ds.roots, b, [this_model, ic_type])

                # Data
                this1 = storage[wave1][region][this_model]

                # Parameters
                parameters = [v.fit_parameter for v in this1.model.variables]
                npar = len(parameters)

                # Mask for the first parameter
                mask1 = mdefine.combined_good_fit_parameter_limit[wave1][region][this_model]
                mask2 = mdefine.is_this_model_preferred(ic_type, ic_limit, this_model)[wave1][region]
                mask_p1 = np.logical_or(mask1, mask2)

                # Second wave
                for iwave2 in range(iwave1+1, len(waves)):

                    # Get the wave
                    wave2 = waves[iwave2]

                    # Data
                    this2 = storage[wave2][region][this_model]

                    # Mask for the second parameter
                    mask1 = mdefine.combined_good_fit_parameter_limit[wave2][region][this_model]
                    mask2 = mdefine.is_this_model_preferred(ic_type, ic_limit, this_model)[wave2][region]
                    mask_p2 = np.logical_or(mask1, mask2)

                    # First parameter
                    for i in range(0, npar):
                        # First parameter name
                        p1_name = parameters[i]

                        # First parameter, label for the plot
                        xlabel = this1.model.variables[i].converted_label + r'$_{%s}$' % wave1

                        # First parameter, data
                        p1 = this1.as_array(p1_name)

                        # First parameter limits
                        p1_limits = limits[p1_name]

                        # Second parameter
                        for j in range(0, npar):
                            # Second parameter name
                            p2_name = parameters[j]

                            # Second parameter, label for the plot
                            ylabel = this2.model.variables[j].converted_label + r'$_{%s}$' % wave2

                            # Second parameter, data
                            p2 = this1.as_array(p2_name)

                            # Second parameter limits
                            p2_limits = limits[p2_name]

                            # Final mask for cross-correlation is the
                            # combination of the first and second parameter
                            # masks
                            final_mask = np.logical_or(mask_p1, mask_p2)

                            # Create the subtitle - model, region, information
                            # on how much of the field of view is not masked,
                            # and the information criterion and limit used.
                            subtitle = dp.concat_string([this_model,
                                                         region,
                                                         dp.get_mask_info_string(final_mask),
                                                         ic_limit_string
                                                         ])

                            # Get the data using the final mask
                            p1 = np.ma.array(p1, mask=final_mask).compressed()
                            p2 = np.ma.array(p2, mask=final_mask).compressed()

                            # Cross correlation statistics
                            r = [spearmanr(p1, p2), pearsonr(p1, p2)]

                            # Form the rank correlation string
                            rstring = 'spr=%1.2f_pea=%1.2f' % (r[0][0], r[1][0])

                            # Plot identity - measurement of the cross
                            # correlation and the physical quantities in the x
                            # and y direction.
                            plot_identity_filename = dp.concat_string([rstring,
                                                                      wave1,
                                                                      p1_name,
                                                                      wave2,
                                                                      p2_name])

                            # Make a scatter plot and the histogram
                            # Make the plot title
                            title = '%s vs. %s \n %s' % (xlabel, ylabel, subtitle)

                            for plot_function in (1, 2):
                                plt.close('all')
                                plt.title(title)
                                plt.xlabel(xlabel)
                                plt.ylabel(ylabel)
                                if plot_function == 1:
                                    plt.scatter(p1, p2)
                                    plot_name = 'scatter'
                                else:
                                    plt.hist2d(p1, p2, bins=bins)
                                    plot_name = 'hist2d'
                                plt.xlim(p1_limits)
                                plt.ylim(p2_limits)
                                x0 = plt.xlim()[0]
                                ylim = plt.ylim()
                                y0 = ylim[0] + 0.3 * (ylim[1] - ylim[0])
                                y1 = ylim[0] + 0.6 * (ylim[1] - ylim[0])
                                plt.text(x0, y0, 'Pearson=%f' % r[0][0], bbox=dict(facecolor=dp.rchi2limitcolor[1], alpha=0.5))
                                plt.text(x0, y1, 'Spearman=%f' % r[1][0], bbox=dict(facecolor=dp.rchi2limitcolor[0], alpha=0.5))
                                if p1_name == p2_name:
                                    plt.plot([p1_limits[0], p1_limits[1]],
                                             [p1_limits[0], p1_limits[1]],
                                             color='r', linewidth=3,
                                             label='%s=%s' % (xlabel, ylabel))
                                final_filename = dp.concat_string([plot_type,
                                                                   plot_identity_filename,
                                                                   subtitle,
                                                                   plot_name]) + '.png'
                                plt.legend(framealpha=0.5)
                                plt.tight_layout()
                                plt.savefig(os.path.join(image, final_filename))
