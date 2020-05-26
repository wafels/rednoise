#
# Analysis - Focusing on the sunspot only:
#            plot spatial distributions of spectral model parameters
#            plot histograms of the spectral model parameters
#
# Assumes that data has been collected and saved in to maps that can be
# easily loaded.
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

# Wavelengths we want to cross correlate
waves = ['131'] #, '171', '193', '211']

# Regions we are interested in
regions = ['six_euv']

# Apodization windows
windows = ['hanning']

# Model results to examine
model_names = ('Power Law + Constant + Lognormal', 'Power Law + Constant')

# Which limit to use
limit_type = 'standard'

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


# Load in all the data
storage = analysis_get_data.get_all_data(waves=waves,
                                         regions=regions,
                                         model_names=model_names)
mdefine = analysis_explore.MaskDefine(storage, limits)


# Get the sunspot outline
sunspot_outline = analysis_get_data.sunspot_outline()


# Plot cross-correlations across different AIA channels
plot_type = 'analysis.within.sunspot'

# Storage for the sunspot submaps
sunspot_submaps = {}

# Plot spatial distributions of the spectral model parameters.
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
                for iwave, wave in enumerate(waves):
                    # Sunspot submaps saving
                    sunspot_submaps[wave] = {}

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

                        # Sunspot submaps saving
                        sunspot_submaps[wave][parameter] = {}

                        # Second parameter limits
                        p1_limits = limits[parameter]

                        # Label
                        label = this.model.variables[i].converted_label

                        # Find out if this model is preferred
                        mask2 = mdefine.is_this_model_preferred(ic_type, ic_limit, this_model)[wave][region]

                        # Final mask combines where the parameters are all nice,
                        # where a good fit was achieved, and where the IC limit
                        # criterion was satisfied.
                        final_mask = np.logical_or(mask1, mask2)

                        image = dp.get_image_model_location(ds.roots, b, [this_model, ic_type])

                        # Create the subtitle - model, region, information
                        # on how much of the field of view is not masked,
                        # and the information criterion and limit used.
                        ic_info_string = '%s, %s' % (ic_limit_string, dp.get_mask_info_string(final_mask))
                        subtitle = dp.concat_string([this_model,
                                                     '%s - %s' % (region, wave),
                                                     ic_info_string,
                                                     limit_type], sep='\n')
                        subtitle_filename = dp.concat_string([this_model,
                                                              region,
                                                              ic_info_string,
                                                              limit_type], sep='.')

                        # Plot identity
                        plot_identity_filename = dp.concat_string([wave, parameter])

                        # Create the masked numpy array
                        map_data = ma.array(p1, mask=final_mask)

                        # Make a SunPy map for nice spatially aware plotting.
                        my_map = analysis_get_data.make_map(submaps['reference region'], map_data)

                        # Get the sunspot
                        polygon, collection = analysis_get_data.rotate_sunspot_outline(sunspot_outline[0], sunspot_outline[1], my_map.date)

                        # Get the bounding box of the sunspot
                        sunspot_bounding_box = False

                        # Get the submap that contains the sunspot
                        sunspot_submap = True

                        # Save each submap for further use
                        sunspot_submaps[wave][parameter] = sunspot_submap

                        #######################################################
                        # Make a spatial distribution map of the spectral model
                        # parameter
                        plt.close('all')
                        # Normalize the color table
                        norm = colors.Normalize(clip=False,
                                                vmin=p1_limits[0].value,
                                                vmax=p1_limits[1].value)

                        # Set up the palette we will use
                        palette = cm.Set2
                        # Bad values are those that are masked out
                        palette.set_bad('white', 1.0)
                        #palette.set_under('green', 1.0)
                        #palette.set_over('red', 1.0)

                        # Begin the plot
                        fig, ax = plt.subplots()
                        # Plot the map
                        ret = my_map.plot(cmap=palette, axes=ax, interpolation='none',
                                          norm=norm)
                        ret.axes.set_title('%s\n%s' % (label, subtitle))
                        ax.add_collection(collection)

                        cbar = fig.colorbar(ret, extend='both', orientation='vertical',
                                            shrink=0.8, label=label)
                        # Fit everything in.
                        ax.autoscale_view()

                        # Dump to file
                        final_filename = dp.concat_string([plot_type,
                                                           'spatial',
                                                           plot_identity_filename,
                                                           subtitle_filename]) + '.png'
                        filepath = os.path.join(image, final_filename)
                        print('Saving to ' + filepath)
                        plt.savefig(filepath)

                        #######################################################
                        # Make a histogram of the spectral model
                        # parameter
                        sunspot_submap_values = np.ma.array(sunspot_submap.data,
                                                        mask=sunspot_submap.mask).compressed()



                        # Dump to file
                        final_filename = dp.concat_string([plot_type,
                                                           'spatial',
                                                           plot_identity_filename,
                                                           subtitle_filename]) + '.png'
                        filepath = os.path.join(image, final_filename)
                        print('Saving to ' + filepath)
                        plt.savefig(filepath)

                # Overplot the histograms from each waveband

