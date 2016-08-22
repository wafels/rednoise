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


# Paper 2
waves = ['94', '131', '171', '193', '211', '335']
regions = ['six_euv']
power_type = 'fourier_power_relative'
limit_type = 'standard'
appended_name = None

# Apodization windows
windows = ['hanning']

# Model results to examine
model_names = ('Power Law + Constant + Lognormal', 'Power Law + Constant')

#
# Details of the analysis
#
limits = da.limits[limit_type]
ic_types = da.ic_details.keys()

#
# Details of the plotting
#
fz = dp.fz
three_minutes = dp.three_minutes
five_minutes = dp.five_minutes
hloc = dp.hloc
linewidth = 3
bins = 50


# Load in all the data
storage = analysis_get_data.get_all_data(waves=waves,
                                         regions=regions,
                                         model_names=model_names,
                                         appended_name=appended_name,
                                         spectral_model='.rnspectralmodels3')
mdefine = analysis_explore.MaskDefine(storage, limits)

# Plot cross-correlations across different AIA channels
plot_type = 'cc.across'

# Different information criteria
for ic_type in ic_types:

    # Get the IC limit
    ic_limits = da.ic_details[ic_type]
    for ic_limit in ic_limits:
        ic_limit_string = '%s>%f' % (ic_type, ic_limit)

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
                    image = dp.get_image_model_location(ds.roots, b, [this_model, ic_limit_string, limit_type])

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
                            p1_data = this1.as_array(p1_name)

                            # First parameter limits
                            p1_limits = limits[p1_name]

                            # Unit
                            if str(this1.model.variables[i].converted_unit) == '':
                                xunit = '(dimensionless)'
                            else:
                                xunit = '(%s)' % str(this1.model.variables[i].converted_unit)

                            # Second parameter
                            for j in range(0, npar):
                                # Second parameter name
                                p2_name = parameters[j]

                                # Second parameter, label for the plot
                                ylabel = this2.model.variables[j].converted_label + r'$_{%s}$' % wave2

                                # Second parameter, data
                                p2_data = this2.as_array(p2_name)

                                # Second parameter limits
                                p2_limits = limits[p2_name]

                                # Unit
                                if str(this2.model.variables[j].converted_unit) == '':
                                    yunit = '(dimensionless)'
                                else:
                                    yunit = '(%s)' % str(this2.model.variables[j].converted_unit)

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
                                p1_compressed = np.ma.array(p1_data, mask=final_mask).compressed().value
                                p2_compressed = np.ma.array(p2_data, mask=final_mask).compressed().value

                                # Limit also by percentage
                                p1_lower_limit, p1_upper_limit, p1_mbld_mask = da.mask_by_limiting_distribution(p1_compressed, [2.5, 97.5])
                                p2_lower_limit, p2_upper_limit, p2_mbld_mask = da.mask_by_limiting_distribution(p2_compressed, [2.5, 97.5])
                                final_mbld_mask = np.logical_or(p1_mbld_mask, p2_mbld_mask)

                                p1 = np.ma.array(p1_compressed, mask=final_mbld_mask).compressed()
                                p2 = np.ma.array(p2_compressed, mask=final_mbld_mask).compressed()

                                # Cross correlation statistics
                                r = [spearmanr(p1, p2), pearsonr(p1, p2)]

                                # Form the rank correlation string
                                rstring = 'spr=%1.2f_pea=%1.2f' % (r[0][0], r[1][0])

                                # Create the subtitle - model, region, information
                                # on how much of the field of view is not masked,
                                # and the information criterion and limit used.
                                number_pixel_string, percent_used_string, mask_info_string = dp.get_mask_info_string(final_mask)
                                ic_info_string = '%s, %s' % (ic_limit_string, mask_info_string)
                                subtitle = dp.concat_string([this_model,
                                                             '%s' % region,
                                                             ic_info_string,
                                                             limit_type], sep='\n')
                                subtitle_filename = dp.concat_string([this_model,
                                                                      region,
                                                                      ic_info_string,
                                                                      limit_type], sep='.')

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
                                #title = '%s vs. %s \n %s' % (xlabel, ylabel, subtitle)
                                title = '%s vs. %s\n%s of all pixels' % (xlabel, ylabel, percent_used_string)

                                for plot_function in (2,):
                                    plt.close('all')
                                    #plt.title(title, fontsize=dp.fontsize)
                                    #plt.xlabel(xlabel, fontsize=dp.fontsize)
                                    #plt.ylabel(ylabel, fontsize=dp.fontsize)
                                    #plt.xlabel('%s %s' % (xlabel, xunit), fontsize=dp.fontsize)
                                    #plt.ylabel('%s %s' % (ylabel, yunit), fontsize=dp.fontsize)
                                    if plot_function == 1:
                                        plt.scatter(p1, p2)
                                        plot_name = 'scatter'
                                    else:
                                        plot_name = 'hist2d'
                                        fig, ax = plt.subplots()
                                        ax.set_title(title, fontsize=dp.fontsize)
                                        ax.set_xlabel(xlabel, fontsize=dp.fontsize)
                                        ax.set_ylabel(ylabel, fontsize=dp.fontsize)
                                        counts, xedges, yedges, _image = ax.hist2d(p1_compressed, p2_compressed, bins=bins,
                                                   range=[p1_limits.value, p2_limits.value])
                                        ax.set_xlim(p1_limits.value)
                                        ax.set_ylim(p2_limits.value)
                                        #plt.xlim([p1_lower_limit, p1_upper_limit])
                                        #plt.ylim([p2_lower_limit, p2_upper_limit])
                                        x0 = plt.xlim()[0]
                                        ylim = plt.ylim()
                                        y0 = ylim[0] + 0.7 * (ylim[1] - ylim[0])
                                        y1 = ylim[0] + 0.9 * (ylim[1] - ylim[0])
                                        ax.text(x0, y0, 'Pearson=%1.2f(%i%%)' % (r[0][0], np.rint(100*r[0][1])), bbox=dict(facecolor=dp.rchi2limitcolor[1], alpha=0.7))
                                        ax.text(x0, y1, 'Spearman=%1.2f(%i%%)' % (r[1][0], np.rint(100*r[1][1])), bbox=dict(facecolor=dp.rchi2limitcolor[0], alpha=0.7))
                                        if p1_name == p2_name:
                                            ax.plot([p1_limits[0].value, p1_limits[1].value],
                                                     [p1_limits[0].value, p1_limits[1].value],
                                                     color='r', linewidth=3,
                                                     label='%s=%s' % (xlabel, ylabel))
                                        # Colorbar
                                        ticks = counts.min() + (counts.max() - counts.min())*np.arange(0, 5)/4.0
                                        cbar_yticklabels = ['%3.1f%%' % (100*tick/counts.sum()) for tick in ticks]
                                        cbar = fig.colorbar(_image, ticks=ticks)
                                        cbar.ax.set_yticklabels(cbar_yticklabels)
                                        cbar.set_label('percentage of population')

                                    final_filename = dp.concat_string([plot_type,
                                                                       plot_identity_filename,
                                                                       subtitle_filename,
                                                                       plot_name,
                                                                       appended_name]) + '.png'
                                    ax.legend(framealpha=0.7, fontsize=dp.fontsize, loc=4)
                                    fig.tight_layout()
                                    print(os.path.join(image, final_filename))
                                    fig.savefig(os.path.join(image, final_filename))
                                    plt.close(fig)
