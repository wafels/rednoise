#
# Analysis - spatial distributions of spectral model parameter values
#
# Show the spatial distributions of spectral model parameter values for a
# list of models.
#
import os
from copy import deepcopy
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import sunpy.map
import analysis_get_data
import details_study as ds
import details_analysis as da summary_statistics, get_mode, limits, get_mask_info, rchi2limitcolor, get_ic_location, get_image_model_location

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


# Get the sunspot outline
sunspot_outline = analysis_get_data.sunspot_outline()


# Plot cross-correlations within the same AIA channel
plot_type = 'spatial.within'

# Different information criteria
for ic_type in ic_types:

    # Get the IC limit
    ic_limit = da.ic_details[ic_type]
    ic_limit_string = '%s>%f' % (ic_type, ic_limit[ic_type])

    # Model name
    for this_model in model_names:

        parameters = this.model.parameters
        npar = len(parameters)

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

                    # Create the subtitle - model, region, information
                    # on how much of the field of view is not masked,
                    # and the information criterion and limit used.
                    subtitle = dp.concat_string([this_model,
                                                 region,
                                                 dp.get_mask_info_string(final_mask),
                                                 ic_limit_string
                                                 ])




# Plot spatial distributions of the spectral model parameters.
for wave in waves:
    for region in regions:

        # branch location
        b = [sd.corename, sd.sunlocation, sd.fits_level, wave, region]

        # Region identifier name
        region_id = sd.datalocationtools.ident_creator(b)

        # Output location
        output = sd.datalocationtools.save_location_calculator(sd.roots, b)["pickle"]
        image = sd.datalocationtools.save_location_calculator(sd.roots, b)["image"]

        # Output filename
        ofilename = os.path.join(output, region_id + '.datacube')

        # Get the region submap
        region_submap = analysis_get_data.get_region_submap(output, region_id)

        for model_name in one_model_name:
            # Get the data for this model
            this = storage[wave][region][model_name]
            # Parameters
            parameters = ("log10(lognormal position)",)
            for parameter in parameters:
                label_index = this.model.parameters.index(parameter)

                # Different information criteria
                for ic_type in ic_limit.keys():
                    image = get_image_model_location(sd.roots, b, [model_name, ic_type])

                    # Where are the good fits
                    mask = this.good_fits()

                    # Data
                    p1 = this.as_array(parameter)

                    # Only consider those pixels where this_model is
                    # preferred by the information criteria
                    mask[get_ic_location(storage[wave][region],
                                         model_name,
                                         model_names,
                                         ic_type=ic_type)] = 1

                    # Apply the limit masks
                    mask[np.where(p1 < limits[parameter][0])] = True
                    mask[np.where(p1 > limits[parameter][1])] = True

                    p1 = 1000 * 4.6296296296296294e-05 * 10.0 ** p1  # mHz

                    # Create the masked numpy array
                    map_data = ma.array(p1, mask=mask)

                    # Make a SunPy map for nice spatially aware plotting.
                    my_map = analysis_get_data.make_map(output, region_id, map_data)
                    my_map = analysis_get_data.hsr2015_map(my_map)

                    # Make a spatial distribution map spectral model parameter
                    plt.close('all')
                    # Normalize the color table
                    norm = colors.Normalize(clip=False, vmin=1.0, vmax=10.0)

                    # Set up the palette we will use
                    palette = cm.Set2
                    palette = cm.Paired
                    # Bad values are those that are masked out
                    palette.set_bad('black', 1.0)
                    #palette.set_under('green', 1.0)
                    #palette.set_over('red', 1.0)
                    # Begin the plot
                    fig, ax = plt.subplots()
                    # Plot the map
                    ret = my_map.plot(cmap=palette, axes=ax, interpolation='none',
                                      norm=norm)
                    #ret.axes.set_title('%s %s %s %s' % (wave, region, this.model.labels[label_index], ic_type))
                    # HSR 2015
                    mhz_label = r"peak frequency (narrow-band oscillation) $\beta$ (mHz)"
                    ret.axes.set_title('%s, %s' % (wave, mhz_label))
                    if region == 'sunspot' or region == 'most_of_fov':
                        ax.add_collection(analysis_get_data.rotate_sunspot_outline(sunspot_outline[0], sunspot_outline[1], my_map.date, edgecolors=['white']))

                    cbar = fig.colorbar(ret, extend='both',
                                        orientation='vertical',
                                        shrink=0.8,
                                        label=mhz_label)
                    # Fit everything in.
                    ax.autoscale_view()

                    # Dump to file
                    filepath = os.path.join(image, '%s.spatial_distrib.%s.%s.%s.png' % (model_name, region_id, parameter, ic_type))
                    print('Saving to ' + filepath)
                    plt.savefig(filepath)
        """
        # Do each measure independently
        for measure in ic_limit.keys():
            # Information Criterion
            this_ic_limit = ic_limit[measure]

            # Get the information for each model
            this0 = storage[wave][region][model_names[0]]
            this1 = storage[wave][region][model_names[1]]

            # Difference in the information criteria
            measure_difference = this0.as_array(measure) - this1.as_array(measure)

            # Go through the common parameters and make a map.
            for parameter in common_parameters:

                # Which label to use
                label_index0 = this0.model.parameters.index(parameter)

                # Zero out the data
                p1 = np.ones_like(this0.as_array('AIC'))

                # Where are the good fits
                good_fit0 = this0.good_fits()
                good_fit1 = this1.good_fits()
                mask = np.ones_like(good_fit0, dtype=bool)

                # Preference for model 0
                model0_where = np.where((measure_difference <= -this_ic_limit) & np.logical_not(good_fit0))

                p1[model0_where] = this0.as_array(parameter)[model0_where]
                mask[model0_where] = False

                # Preference for model 1
                model1_where = np.where((measure_difference >= this_ic_limit) & np.logical_not(good_fit1))

                p1[model1_where] = this1.as_array(parameter)[model1_where]
                mask[model1_where] = False

                # Apply the limit masks
                # For HSR 2015 mask[np.where(p1 < limits[parameter][0])] = True
                # For HSR 2015 mask[np.where(p1 > limits[parameter][1])] = True

                # Make a SunPy map for nice spatially aware plotting.
                map_data = ma.array(p1, mask=mask)

                my_map = sunpy.map.Map(map_data, deepcopy(region_submap.meta))
                my_map = analysis_get_data.hsr2015_map(my_map)

                # Make a spatial distribution map spectral model parameter
                plt.close('all')
                # Normalize the color table
                norm = colors.Normalize(clip=False,
                                        vmin=limits[parameter][0],
                                        vmax=limits[parameter][1])

                # Set up the palette we will use
                palette = cm.Set2
                # Bad values are those that are masked out
                palette.set_bad('black', 1.0)
                #palette.set_under('green', 1.0)
                #palette.set_over('red', 1.0)
                # Begin the plot
                fig, ax = plt.subplots()
                # Plot the map
                ret = my_map.plot(cmap=palette, axes=ax, interpolation='none',
                                  norm=norm)
                #ret.axes.set_title('across models %s %s %s %s %f' % (wave, region, this0.model.labels[label_index0], measure, this_ic_limit))
                ret.axes.set_title("%s, cross-model parameter %s " % (wave, this0.model.labels[label_index0]))

                if region == 'most_of_fov':
                    ax.add_collection(analysis_get_data.rotate_sunspot_outline(sunspot_outline[0], sunspot_outline[1], my_map.date, edgecolors=['k']))

                cbar = fig.colorbar(ret, extend='both', orientation='vertical',
                                    shrink=0.8, label=this0.model.labels[label_index0])
                # Fit everything in.
                ax.autoscale_view()

                # Dump to file
                filepath = os.path.join(image, 'across_models.spatial_distrib.across_models.%s.%s.%s.%f.png' % (region_id, parameter, measure, this_ic_limit))
                print('Saving to ' + filepath)
                plt.savefig(filepath)

                # Save a histogram of the results
                # Summary stats

                # Apply the limit masks
                mask[np.where(p1 < limits[parameter][0])] = True
                mask[np.where(p1 > limits[parameter][1])] = True
                map_data = ma.array(p1, mask=mask)
                pm1 = map_data.compressed()

                # Store the data for the next set of plots
                storage_common_parameter[wave][region][measure][parameter] = map_data

                # Summary statistics
                ss = summary_statistics(pm1)

                # Label for the plots
                label1 = this0.model.labels[label_index0]

                # Identifier of the plot
                plot_identity = wave + '.' + region + '.' + parameter + '.%s.%f' % (measure, this_ic_limit)

                # Title of the plot
                title = plot_identity + get_mask_info(mask)

                plt.close('all')
                plt.figure(1, figsize=(10, 10))
                for ibinning, binning in enumerate(hloc):
                    plt.subplot(len(hloc), 1, ibinning+1)
                    h_info = hist(pm1, bins=binning)
                    mode = get_mode(h_info)
                    plt.axvline(ss['mean'], color='r', label='mean=%f' % ss['mean'], linewidth=linewidth)
                    plt.axvline(mode[1][0], color='g', label='%f<mode<%f' % (mode[1][0], mode[1][1]), linewidth=linewidth)
                    plt.axvline(mode[1][1], color='g', linewidth=linewidth)
                    plt.xlabel(label1)
                    plt.title(str(binning) + ' : %s\n across models' % (title,))
                    plt.legend(framealpha=0.5, fontsize=8)
                    plt.xlim(limits[parameter])

                plt.tight_layout()
                ofilename = 'across_model.' + plot_identity + '.hist.png'
                plt.savefig(os.path.join(image, ofilename))

#
# Make a power-law index difference map
#


# Go through all the common parameters and make 2-d histograms of them across
# wavelengths
for region in regions:
    for measure in ic_limit.keys():
        for parameter in common_parameters:

            # Which label to use
            label_index = this0.model.parameters.index(parameter)
            p_label = this0.model.labels[label_index]

            for iwave1, wave1 in enumerate(waves):
                map_data1 = storage_common_parameter[wave1][region][measure][parameter]

                # Get the mask out of the map data
                mask1 = np.ma.getmask(map_data1)

                # branch location
                b = [sd.corename, sd.sunlocation, sd.fits_level, wave1, region]

                # Region identifier name
                region_id = sd.datalocationtools.ident_creator(b)

                # Output location
                output = sd.datalocationtools.save_location_calculator(sd.roots, b)["pickle"]
                image = sd.datalocationtools.save_location_calculator(sd.roots, b)["image"]

                # Output filename
                ofilename = os.path.join(output, region_id + '.datacube')

                for iwave2 in range(iwave1+1, len(waves)):
                    wave2 = waves[iwave2]
                    map_data2 = storage_common_parameter[wave2][region][measure][parameter]

                    # Get the masks out of the map data
                    mask2 = np.ma.getmask(map_data2)

                    # Combine the masks
                    combined_mask = np.logical_not(np.logical_not(mask1) * np.logical_not(mask2))

                    # Get the two parameters
                    p1 = np.ma.array(np.ma.getdata(map_data1), mask=combined_mask).compressed()
                    p2 = np.ma.array(np.ma.getdata(map_data2), mask=combined_mask).compressed()

                    # Cross correlation statistics
                    r = [spearmanr(p1, p2), pearsonr(p1, p2)]

                    # Form the rank correlation string
                    rstring = 'spr=%1.2f_pea=%1.2f' % (r[0][0], r[1][0])

                    # Identifier of the plot
                    plot_identity = rstring + '.' + region + '.' + wave1 + '.' + wave2 + '.' + parameter + '.' + measure

                    # Make a scatter plot
                    xlabel = p_label + r'$_{%s}$' % wave1
                    ylabel = p_label + r'$_{%s}$' % wave2
                    title = '%s vs. %s, \n%s' % (xlabel, ylabel, plot_identity)
                    plt.close('all')
                    plt.title(title)
                    plt.xlabel(xlabel)
                    plt.ylabel(ylabel)
                    plt.scatter(p1, p2)
                    plt.xlim(limits[parameter])
                    plt.ylim(limits[parameter])
                    x0 = plt.xlim()[0]
                    ylim = plt.ylim()
                    y0 = ylim[0] + 0.3 * (ylim[1] - ylim[0])
                    y1 = ylim[0] + 0.6 * (ylim[1] - ylim[0])
                    plt.text(x0, y0, 'Pearson=%f' % r[0][0], bbox=dict(facecolor=rchi2limitcolor[1], alpha=0.5))
                    plt.text(x0, y1, 'Spearman=%f' % r[1][0], bbox=dict(facecolor=rchi2limitcolor[0], alpha=0.5))
                    plt.plot([limits[parameter][0], limits[parameter][1]],
                             [limits[parameter][0], limits[parameter][1]],
                             color='r', linewidth=3,
                             label='%s=%s' % (xlabel, ylabel))
                    ofilename = 'across_model.' + plot_identity + '.scatter.png'
                    plt.legend(framealpha=0.5)
                    plt.tight_layout()
                    plt.savefig(os.path.join(image, ofilename))

                    # Make a 2d histogram
                    plt.close('all')
                    plt.title(title)
                    plt.xlabel(xlabel)
                    plt.ylabel(ylabel)
                    plt.hist2d(p1, p2, bins=bins_2d, range=[limits[parameter], limits[parameter]])
                    x0 = plt.xlim()[0]
                    ylim = plt.ylim()
                    y0 = ylim[0] + 0.3 * (ylim[1] - ylim[0])
                    y1 = ylim[0] + 0.6 * (ylim[1] - ylim[0])
                    plt.text(x0, y0, 'Pearson=%f' % r[0][0], bbox=dict(facecolor=rchi2limitcolor[1], alpha=0.5))
                    plt.text(x0, y1, 'Spearman=%f' % r[1][0], bbox=dict(facecolor=rchi2limitcolor[0], alpha=0.5))
                    plt.plot([limits[parameter][0], limits[parameter][1]],
                             [limits[parameter][0], limits[parameter][1]],
                             color='r', linewidth=3,
                             label="%s=%s" % (xlabel, ylabel))
                    ofilename = 'across_model.' + plot_identity + '.hist2d.png'
                    plt.legend(framealpha=0.5)
                    plt.tight_layout()
                    plt.savefig(os.path.join(image, ofilename))
"""