#
# Analysis - spatial distributions of spectral model parameters that are common
# across a number of models, as defined by the user.  At each pixel the model
# selected by the information criterion is selected, and its value is plotted.
#
#
# TODO: Nverplot the simulated data power law index histograms for all 3 simulations.
# TODO: Note the number of pixels that were fit in the plot.
# TODO: Normalize the histograms to the number of pixels that were fit.
#
import os
import pickle
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import analysis_get_data
import analysis_explore
import details_study as ds
import details_analysis as da
import details_plots as dp
from tools import statistics_tools

import matplotlib.cm as cm


# Paper 2
# Wavelengths we want to cross correlate
waves = ['94', '131', '171', '193', '211', '335', ]
waves = ['171', '193']
waves = ['171']
regions = ['six_euv']
power_type = 'fourier_power_relative'
limit_type = 'standard'

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
bins = 100
fontsize = dp.fontsize

# Load in all the data
storage = analysis_get_data.get_all_data(waves=waves,
                                         regions=regions,
                                         model_names=model_names,
                                         spectral_model='.rnspectralmodels3')
# Define the masks
mdefine = analysis_explore.MaskDefine(storage, limits)

# Get the common parameters
parameters = mdefine.common_parameters
npar = len(parameters)

# Get the sunspot outline
sunspot_outline = analysis_get_data.sunspot_outline()

# Plot spatial distributions of the common parameters
plot_type = 'spatial.common'

# Storage for the file paths
filepaths_root = ds.datalocationtools.save_location_calculator(ds.roots, [ds.corename, ds.sunlocation, ds.fits_level])['pickle']
filepaths = []



# Plot spatial distributions of the spectral model parameters.
# Different information criteria
for ic_type in ic_types:

    # Get the IC limit
    ic_limits = da.ic_details[ic_type]
    for ic_limit in ic_limits:
        ic_limit_string = '%s.ge.%f' % (ic_type, ic_limit)

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
                if iwave == 0 and not (ds.corename in ds.simulation):
                    all_submaps = analysis_get_data.get_region_submap(output, region_id)
                    region_submap = all_submaps['reference region']

                # Get preferred model index.  This is an index to the list of
                # models used.  If the mask value is true, then the preferred
                # model did NOT exceed the information criterion limit.
                preferred_model_index = mdefine.which_model_is_preferred(ic_type, ic_limit)[wave][region]

                # Go through the common parameters and make a map.
                for parameter in parameters:

                    # Go through all the models and construct
                    final_data = np.zeros_like(np.ma.getmask(preferred_model_index), dtype=float)
                    final_mask = np.ones_like(final_data, dtype=bool)

                    # Parameter limits
                    p1_limits = limits[parameter]

                    # Construct the map by looking at each model individually.
                    # The values and the
                    for this_model_index in range(0, len(model_names)):
                        # Get the model name
                        this_model = mdefine.available_models[this_model_index]

                        # Find out where this model is preferred
                        this_model_mask = np.ma.getdata(preferred_model_index) == this_model_index

                        # Find out where the model exceeds the information
                        # criterion
                        exceeds_ic_mask = np.ma.getmask(preferred_model_index)

                        # Good fit for this model?
                        cgfp_mask = mdefine.combined_good_fit_parameter_limit[wave][region][this_model]

                        # Combined final mask for this model
                        combined_mask = np.logical_or(np.logical_or(np.logical_not(this_model_mask), exceeds_ic_mask), cgfp_mask)

                        # Get the data for this model
                        this = storage[wave][region][this_model]

                        # Apply the combined mask to get the data
                        these_data = np.where(np.logical_not(combined_mask))
                        final_data[these_data] = this.as_array(parameter)[these_data]

                        # Update the final mask
                        final_mask[these_data] = False

                    # Label
                    fit_parameters = [v.fit_parameter for v in this.model.variables]
                    parameter_index = fit_parameters.index(parameter)
                    label = this.model.variables[parameter_index].converted_label
                    fit_parameter = this.model.variables[parameter_index].fit_parameter

                    image = dp.get_image_model_location(ds.roots, b, [this_model, ic_limit_string, limit_type])

                    # Create the subtitle - model, region, information
                    # on how much of the field of view is not masked,
                    # and the information criterion and limit used.
                    number_pixel_string, percent_used_string, mask_info_string = dp.get_mask_info_string(final_mask)
                    ic_info_string = ''  # ''%s, %s' % (ic_limit_string, mask_info_string)
                    subtitle_filename = dp.concat_string([region,
                                                          limit_type], sep='.')

                    # Plot identity
                    plot_identity_filename = dp.concat_string([wave, parameter])

                    # Create the masked numpy array
                    map_data = ma.array(final_data, mask=final_mask)

                    # Range
                    print('{:s}: {:n}->{:n}'.format(parameter, np.nanmin(map_data), np.nanmax(map_data)))

                    if ds.corename in ds.simulation:
                        plt.imshow(np.transpose(map_data), cmap=cm.Set2, origin='lower')
                        plt.xlabel('x (pixels)', fontsize=fontsize)
                        plt.ylabel('y (pixels)', fontsize=fontsize)
                        cbar = plt.colorbar()
                        cbar.ax.set_ylabel(label, fontsize=fontsize)
                        map_title1 = ds.sim_name[ds.corename]
                        map_title2 = 'simulated AIA ' + wave + ' Angstrom, {:s} fit'.format(percent_used_string)
                        map_title3 = fit_parameter + ' ' + label
                        map_title = map_title1 + '\n' + map_title2 + '\n' + map_title3
                        plt.title(map_title, fontsize=fontsize)
                        final_filename = dp.concat_string([plot_type,
                                                           plot_identity_filename,
                                                           subtitle_filename]).replace(' ', '')
                        filepath = os.path.join(image, final_filename + '.{:s}.png'.format(ds.corename))
                        print('Saving to ' + filepath)
                        plt.savefig(filepath, bbox_inches='tight')
                        plt.close('all')

                        bins = 50
                        ss = statistics_tools.Summary(map_data.compressed())
                        md = map_data.compressed()
                        weights = np.ones_like(md)/len(md)
                        plt.hist(md, bins=bins, weights=weights)
                        plt.title(map_title)
                        plt.xlabel(label, fontsize=fontsize)
                        plt.ylabel('number', fontsize=fontsize)
                        filepath = os.path.join(image, final_filename + '.{:s}.distribution.png'.format(ds.corename))
                        print('Saving to ' + filepath)
                        plt.savefig(filepath, bbox_inches='tight')
                        plt.close('all')

                        subtitle = dp.concat_string(['%s - %s' % (region, wave),
                                                    percent_used_string], sep='\n')
                        my_map = map_data
                    else:
                        # Make a SunPy map for nice spatially aware plotting.
                        my_map = analysis_get_data.make_map(region_submap, map_data)

                        # Get the sunspot
                        polygon, sunspot_collection = analysis_get_data.rotate_sunspot_outline(sunspot_outline[0],
                                                                                      sunspot_outline[1],
                                                                                      my_map.date,
                                                                                      edgecolors=[dp.sunspot_outline.color])

                        subtitle = dp.concat_string(['%s - %s' % (region, wave),
                                                    percent_used_string], sep='\n')
                        # Make a spatial distribution map spectral model parameter
                        plt.close('all')
                        # Normalize the color table
                        norm = colors.Normalize(clip=False,
                                                vmin=p1_limits[0].value,
                                                vmax=p1_limits[1].value)

                        # Set up the palette we will use
                        palette = dp.spectral_parameters.cm
                        # Bad values are those that are masked out
                        palette.set_bad(dp.spectral_parameters.bad, 1.0)
                        #palette.set_under('green', 1.0)
                        #palette.set_over('red', 1.0)

                        # Begin the plot
                        fig, ax = plt.subplots()
                        # Plot the map
                        ret = my_map.plot(cmap=palette, axes=ax, interpolation='none', norm=norm)
                        #ret.axes.set_title('%s\n%s' % (label, subtitle))
                        title = label + r'$_{%s}$' % wave
                        #ret.axes.set_title(title + '\n[%s]' % percent_used_string, fontsize=fontsize)
                        #map_title = title + '\n%s of all pixels' % percent_used_string
                        map_title = 'power law index ' + label
                        map_title += '\nAIA ' + wave + r'$\mathrm{\AA}$' + ', {:s} fit'.format(percent_used_string)
                        ret.axes.set_title(map_title, fontsize=fontsize)
                        #X = my_map.xrange[0].value + my_map.scale.x.value * np.arange(0, my_map.dimensions.x.value)
                        #Y = my_map.yrange[0].value + my_map.scale.y.value * np.arange(0, my_map.dimensions.y.value)
                        #ret.axes.contour(X, Y, all_submaps[submap_type].data, 3,
                        #                 colors=["#0088ff", "#0044ff", "#0000ff"],
                        #                 linewidths=[2, 2, 2])
                        ax.add_collection(sunspot_collection)
                        cbar = fig.colorbar(ret, extend='both', orientation='vertical', shrink=0.8, label=label)

                        # Fit everything in.
                        ax.autoscale_view()

                        # Dump to file
                        #final_filename = dp.concat_string([plot_type,
                        #                                   submap_type,
                        #                                   plot_identity_filename,
                        #                                   subtitle_filename])
                        final_filename = dp.concat_string([plot_type,
                                                           plot_identity_filename,
                                                           subtitle_filename]).replace(' ', '')
                        filepath = os.path.join(image, final_filename + '.png')
                        print('Saving to ' + filepath)
                        plt.savefig(filepath, bbox_inches='tight')
                        plt.close('all')

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
                        filepath = os.path.join(image, final_filename + '.distribution.png')
                        print('Saving to ' + filepath)
                        plt.savefig(filepath, bbox_inches='tight')
                        plt.close('all')


                    # Save the map to a file for later use
                    final_pickle_filepath = os.path.join(output,
                                                         final_filename + '.pkl')
                    print('Saving map to %s' % final_pickle_filepath)
                    f = open(final_pickle_filepath, 'wb')
                    pickle.dump(map_title, f)
                    pickle.dump(subtitle, f)
                    pickle.dump(my_map, f)
                    f.close()

                    # Store the location and file name
                    print('Adding %s' % final_pickle_filepath)
                    filepaths.append(final_pickle_filepath)

# Save the location and file name data
filepaths_filepath = os.path.join(filepaths_root, "analysis_spatial_distribution_common_spectral_model_parameters.filepaths.pkl")
f = open(filepaths_filepath, 'wb')
pickle.dump(filepaths, f)
f.close()

