#
# Analysis - Plot the spatial distributions of spectral model parameters
#
import os
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

import analysis_get_data
import analysis_explore
import details_study as ds
import details_analysis as da
import details_plots as dp
from tools import statistics_tools

# Wavelengths we want to cross correlate
waves = ['94', '131', '171', '193', '211', '335']
regions = ['six_euv']
power_type = 'fourier_power_relative'
limit_type = 'standard'

# Apodization windows
windows = ['hanning']

# Model results to examine
model_names = ('Power Law + Constant', 'Power Law + Constant + Lognormal')


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
linewidth = dp.linewidth
bins = dp.histogram_1d_bins
fontsize = dp.fontsize

# Load in all the data
storage = analysis_get_data.get_all_data(waves=waves,
                                         regions=regions,
                                         model_names=model_names,
                                         spectral_model='.rnspectralmodels3')
mdefine = analysis_explore.MaskDefine(storage, limits)


# Get the sunspot outline
sunspot_outline = analysis_get_data.sunspot_outline()


# Plot cross-correlations across different AIA channels
plot_type = 'spatial.within'

# Plot spatial distributions of the spectral model parameters.
# Different information criteria
for ic_type in ic_types:

    # Get the IC limit
    ic_limits = da.ic_details[ic_type]
    for ic_limit in ic_limits:
        ic_limit_string = '%s.ge.%f' % (ic_type, ic_limit)

        # Model name
        for this_model in model_names:

            # Select a region
            for region in regions:

                # First wave
                for iwave, wave in enumerate(waves):

                    # branch location
                    b = [ds.corename, ds.sunlocation, ds.fits_level, wave, region]

                    # Region identifier name
                    region_id = ds.datalocationtools.ident_creator(b)

                    # Output location
                    output = ds.datalocationtools.save_location_calculator(ds.roots, b)["pickle"]
                    image = ds.datalocationtools.save_location_calculator(ds.roots, b)["image"]

                    # Output filename
                    ofilename = os.path.join(output, region_id + '.datacube')

                    # Get the region submap
                    if iwave == 0:
                        submaps = analysis_get_data.get_region_submap(output, region_id)

                    # Get the data for this model
                    this = storage[wave][region][this_model]

                    # Parameters
                    parameters = [v.fit_parameter for v in this.model.variables]
                    npar = len(parameters)

                    # Get the combined mask
                    mask1 = mdefine.combined_good_fit_parameter_limit[wave][region][this_model]

                    for i in range(0, npar):
                        # Data
                        parameter = parameters[i]
                        p1 = this.as_array(parameter)

                        # Second parameter limits
                        p1_limits = limits[parameter]

                        # Label
                        label = this.model.variables[i].converted_label  # + r'$_{%s}$' % wave

                        # Find out if this model is preferred
                        mask2 = mdefine.is_this_model_preferred(ic_type, ic_limit, this_model)[wave][region]

                        # Final mask combines where the parameters are all nice,
                        # where a good fit was achieved, and where the IC limit
                        # criterion was satisfied.
                        final_mask = np.logical_or(mask1, mask2)

                        image = dp.get_image_model_location(ds.roots, b, [this_model, ic_limit_string, limit_type])

                        # Create the subtitle - model, region, information
                        # on how much of the field of view is not masked,
                        # and the information criterion and limit used.
                        number_pixel_string, percent_used_string, mask_info_string = dp.get_mask_info_string(final_mask)
                        ic_info_string = '%s, %s' % (ic_limit_string, mask_info_string)
                        subtitle = dp.concat_string([this_model,
                                                     '%s - %s' % (region, wave),
                                                     ic_info_string,
                                                     limit_type], sep='\n')
                        subtitle_filename = dp.concat_string([this_model,
                                                              region,
                                                              ic_limit_string,
                                                              limit_type], sep='.')

                        # Plot identity
                        plot_identity_filename = dp.concat_string([wave, parameter])

                        # Create the masked numpy array
                        map_data = ma.array(p1, mask=final_mask)

                        # Make a SunPy map for nice spatially aware plotting.
                        my_map = analysis_get_data.make_map(submaps['reference region'], map_data)

                        # Get the sunspot

                        # Get the sunspot
                        polygon, sunspot_collection = analysis_get_data.rotate_sunspot_outline(sunspot_outline[0],
                                                                                  sunspot_outline[1],
                                                                                  my_map.date,
                                                                                  edgecolors=[dp.spatial_plots['sunspot outline']])

                        # Make a spatial distribution map spectral model
                        # parameter
                        plt.close('all')
                        # Normalize the color table
                        norm = colors.Normalize(clip=False,
                                                vmin=p1_limits[0].value,
                                                vmax=p1_limits[1].value)

                        # Set up the palette we will use
                        palette = dp.spatial_plots['color table']
                        # Bad values are those that are masked out
                        palette.set_bad(dp.spatial_plots['bad value'], 1.0)
                        #palette.set_under('green', 1.0)
                        #palette.set_over('red', 1.0)

                        # Begin the plot
                        fig, ax = plt.subplots()
                        # Plot the map
                        ret = my_map.plot(cmap=palette, axes=ax, interpolation='none',
                                          norm=norm)
                        #ret.axes.set_title('%s\n%s' % (label, subtitle))
                        title = '%s\n%s of all pixels' % (label, percent_used_string)
                        title = label + '\n' + 'AIA ' + wave + ' Angstrom, {:s} fit'.format(percent_used_string)
                        ret.axes.set_title(title, fontsize=fontsize)
                        ax.add_collection(sunspot_collection)

                        cbar = fig.colorbar(ret, extend='both', orientation='vertical',
                                            shrink=0.8, label=label)
                        # Fit everything in.
                        ax.autoscale_view()

                        # Dump to file
                        final_filename = dp.concat_string([plot_type,
                                                           plot_identity_filename,
                                                           subtitle_filename]).replace(' ', '') + '.eps'
                        filepath = os.path.join(image, final_filename)
                        print('Saving to ' + filepath)
                        plt.savefig(filepath, bbox_inches='tight')
                        plt.close(fig)

                        #
                        # Distributions
                        #
                        bins = 50
                        ss = statistics_tools.Summary(map_data.compressed())
                        plt.hist(map_data.compressed(), bins=bins)
                        plt.title(title)
                        plt.xlabel(parameter)
                        plt.ylabel('number')
                        plt.axvline(ss.mean,
                                    label='mean [{:n}]'.format(ss.mean),
                                    linestyle=dp.mean.linestyle,
                                    color=dp.mean.color)
                        plt.axvline(ss.median,
                                    label='median [{:n}]'.format(ss.median),
                                    linestyle=dp.median.linestyle,
                                    color=dp.median.color)
                        plt.axvline(ss.percentile[0],
                                    label='2.5% [{:n}]'.format(ss.percentile[0]),
                                    linestyle=dp.percentile0.linestyle,
                                    color=dp.percentile0.color)
                        plt.axvline(ss.percentile[1],
                                    label='97.5% [{:n}]'.format(ss.percentile[1]),
                                    linestyle=dp.percentile1.linestyle,
                                    color=dp.percentile1.color)
                        plt.legend(loc=1, framealpha=0.5)
                        filepath = os.path.join(image, final_filename + '.distribution.eps')
                        print('Saving to ' + filepath)
                        plt.savefig(filepath, bbox_inches='tight')
                        plt.close('all')
