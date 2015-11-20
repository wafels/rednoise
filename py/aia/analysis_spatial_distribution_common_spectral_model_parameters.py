#
# Analysis - spatial distributions of spectral model parameters that are common
# across a number of models, as defined by the user.  At each pixel the model
# selected by the information criterion is selected, and its value is plotted.
#
import os
import cPickle as pickle
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import analysis_get_data
import analysis_explore
import details_study as ds
import details_analysis as da
import details_plots as dp


# Wavelengths we want to cross correlate
waves = ['131', '171', '193', '211', '335', '94']

# Regions we are interested in
# regions = ['sunspot', 'moss', 'quiet Sun', 'loop footpoints']
# regions = ['most_of_fov']
regions = ['six_euv']


# Apodization windows
windows = ['hanning']

# Model results to examine
model_names = ('Power Law + Constant + Lognormal', 'Power Law + Constant')

#
# Details of the analysis
#
limits = da.limits['standard']
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
# Define the masks
mdefine = analysis_explore.MaskDefine(storage, limits)

# Get the common parameters
parameters = mdefine.common_parameters
npar = len(parameters)

# Get the sunspot outline
sunspot_outline = analysis_get_data.sunspot_outline()

# Plot spatial distributions of the common parameters
plot_type = 'spatial.common'

# Plot spatial distributions of the spectral model parameters.
# Different information criteria
for ic_type in ic_types:

    # Get the IC limit
    ic_limits = da.ic_details[ic_type]
    for ic_limit in ic_limits:
        ic_limit_string = '%s>%f' % (ic_type, ic_limit)

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

                    image = dp.get_image_model_location(ds.roots, b, [this_model, ic_type])

                    # Create the subtitle - model, region, information
                    # on how much of the field of view is not masked,
                    # and the information criterion and limit used.
                    ic_info_string = '%s, %s' % (ic_limit_string, dp.get_mask_info_string(final_mask))
                    subtitle_filename = dp.concat_string([region,
                                                          ic_info_string], sep='.')

                    # Plot identity
                    plot_identity_filename = dp.concat_string([wave, parameter])

                    # Create the masked numpy array
                    map_data = ma.array(final_data, mask=final_mask)

                    # Make a SunPy map for nice spatially aware plotting.
                    my_map = analysis_get_data.make_map(region_submap, map_data)

                    for submap_type in all_submaps.keys():

                        # Get the sunspot
                        sunspot_collection = analysis_get_data.rotate_sunspot_outline(sunspot_outline[0],
                                                                                  sunspot_outline[1],
                                                                                  my_map.date,
                                                                                  edgecolors=[dp.spatial_plots['sunspot outline']])

                        subtitle = dp.concat_string(['%s - %s' % (region, wave),
                                                    ic_info_string,
                                                    submap_type], sep='\n')
                        # Make a spatial distribution map spectral model parameter
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
                        ret = my_map.plot(cmap=palette, axes=ax, interpolation='none', norm=norm)
                        ret.axes.set_title('%s\n%s' % (label, subtitle))
                        X = my_map.xrange[0].value + my_map.scale.x.value * np.arange(0, my_map.dimensions.x.value)
                        Y = my_map.yrange[0].value + my_map.scale.y.value * np.arange(0, my_map.dimensions.y.value)
                        ret.axes.contour(X, Y, all_submaps[submap_type].data, 3,
                                         colors=["#0088ff", "#0044ff", "#0000ff"],
                                         linewidths=[2, 2, 2])
                        ax.add_collection(sunspot_collection)
                        cbar = fig.colorbar(ret, extend='both', orientation='vertical', shrink=0.8, label=label)

                        # Fit everything in.
                        ax.autoscale_view()

                        # Dump to file
                        final_filename = dp.concat_string([plot_type,
                                                           submap_type,
                                                           plot_identity_filename,
                                                           subtitle_filename])
                        filepath = os.path.join(image, final_filename + '.png')
                        print('Saving to ' + filepath)
                        plt.savefig(filepath)

                        # Save the map to a file for later use
                        final_pickle_filepath = os.path.join(output,
                                                             final_filename + '.pkl')
                        print('Saving map to %s' % final_pickle_filepath)
                        f = open(final_pickle_filepath, 'wb')
                        pickle.dump(my_map, f)
                        f.close()
