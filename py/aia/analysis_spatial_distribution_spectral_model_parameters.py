#
# Analysis - Plot the spatial distributions of spectral model parameters
#
import os
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from astroML.plotting import hist
import analysis_get_data
import study_details as sd
from analysis_details import summary_statistics, get_mode, limits, get_mask_info

# Wavelengths we want to analyze
waves = ['193']

# Regions we are interested in
regions = ['moss', 'sunspot', 'quiet Sun', 'loop footpoints']
regions = ['most_of_fov']
# Apodization windows
windows = ['hanning']

# Model results to examine
model_names = ('Power law + Constant + Lognormal', 'Power law + Constant')\

# Number of bins
hloc = (100,)
linewidth = 3

# IC
ic_types = ('none',)
ic_limit = 10.0

# Load in all the data
storage = analysis_get_data.get_all_data(waves=waves, regions=regions)

# Get the sunspot outline
sunspot_outline = analysis_get_data.sunspot_outline()

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

        for model_name in model_names:
            # Get the data for this model
            this = storage[wave][region][model_name]
            """
            # Parameters
            parameters = this.model.parameters
            for parameter in parameters:
                label_index = this.model.parameters.index(parameter)

                # Different information criteria
                for ic_type in ic_types:
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

                    # Create the masked numpy array
                    map_data = ma.array(p1, mask=mask)
                    # Make a SunPy map for nice spatially aware plotting.
                    my_map = analysis_get_data.make_map(output, region_id, map_data)

                    # Make a spatial distribution map spectral model parameter
                    plt.close('all')
                    # Normalize the color table
                    norm = colors.Normalize(clip=False, vmin=limits[parameter][0], vmax=limits[parameter][1])

                    # Set up the palette we will use
                    palette = cm.Set2
                    # Bad values are those that are masked out
                    palette.set_bad('black', 1.0)
                    palette.set_under('green', 1.0)
                    palette.set_over('red', 1.0)
                    # Begin the plot
                    fig, ax = plt.subplots()
                    # Plot the map
                    ret = my_map.plot(cmap=palette, axes=ax, interpolation='none',
                                      norm=norm)
                    ret.axes.set_title('%s %s %s %s' % (wave, region, this.model.labels[label_index], ic_type))
                    if region == 'sunspot':
                        ax.add_collection(analysis_get_data.rotate_sunspot_outline(sunspot_outline[0], sunspot_outline[1], my_map.date))

                    cbar = fig.colorbar(ret, extend='both', orientation='horizontal',
                                        shrink=0.8, label=this.model.labels[label_index])
                    # Fit everything in.
                    ax.autoscale_view()

                    # Dump to file
                    filepath = os.path.join(image, '%s.spatial_distrib.%s.%s.%s.png' % (model_name, region_id, parameter, ic_type))
                    print('Saving to ' + filepath)
                    plt.savefig(filepath)
            """
        par1 = storage[wave][region][model_names[0]].model.parameters
        par2 = storage[wave][region][model_names[1]].model.parameters
        common_parameters = set(par1).intersection(par2)
        # Do each measure independently
        for measure in ("AIC", "BIC"):
            # Get the information for each model
            this0 = storage[wave][region][model_names[0]]
            this1 = storage[wave][region][model_names[1]]

            # Where are the good fits
            good_fit0 = this0.good_fits()
            good_fit1 = this1.good_fits()
            good_fit_both = np.logical_not(np.logical_or(np.logical_not(good_fit0), np.logical_not(good_fit1)))
            # Difference in the information criteria
            measure_difference = this0.as_array(measure) - this1.as_array(measure)

            # Go through the common parameters and make a map.
            for parameter in common_parameters:
                # Zero out the data
                p1 = np.zeros_like(this.as_array('AIC'))
                good_fit_both = np.zeros_like(p1, dtype=bool)

                label_index0 = this0.model.parameters.index(parameter)

                # Model 0
                model0_where = np.where(measure_difference < -ic_limit)
                p1[model0_where] = this0.as_array(parameter)[model0_where]

                # Model 1
                model1_where = np.where(measure_difference > ic_limit)
                p1[model1_where] = this1.as_array(parameter)[model1_where]

                # Make a SunPy map for nice spatially aware plotting.
                map_data = ma.array(p1, mask=good_fit_both)
                my_map = analysis_get_data.make_map(output, region_id, map_data)

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
                palette.set_under('green', 1.0)
                palette.set_over('red', 1.0)
                # Begin the plot
                fig, ax = plt.subplots()
                # Plot the map
                ret = my_map.plot(cmap=palette, axes=ax, interpolation='none',
                                  norm=norm)
                ret.axes.set_title('across models %s %s %s %s' % (wave, region, this0.model.labels[label_index0], measure))
                if region == 'sunspot':
                    ax.add_collection(analysis_get_data.rotate_sunspot_outline(sunspot_outline[0], sunspot_outline[1], my_map.date))

                cbar = fig.colorbar(ret, extend='both', orientation='horizontal',
                                    shrink=0.8, label=this.model.labels[label_index0])
                # Fit everything in.
                ax.autoscale_view()

                # Dump to file
                filepath = os.path.join(image, 'across_models.spatial_distrib.across_models.%s.%s.%s.png' % (region_id, parameter, measure))
                print('Saving to ' + filepath)
                plt.savefig(filepath)

                # Save a histogram of the results
                # Summary stats

                # Apply the limit masks
                good_fit_both[np.where(p1 < limits[parameter][0])] = 1
                good_fit_both[np.where(p1 > limits[parameter][1])] = 1
                map_data = ma.array(p1, mask=good_fit_both)
                pm1 = map_data.compressed()

                # Store the data for the next set of plots
                #storage_common_parameter[wave][region][measure][parameter] = map_data

                # Summary statistics
                ss = summary_statistics(pm1)

                # Label for the plots
                label1 = this0.model.labels[label_index0]

                # Identifier of the plot
                plot_identity = wave + '.' + region + '.' + parameter + '.' + measure

                # Title of the plot
                title = plot_identity + get_mask_info(good_fit_both)

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

# Go through all the common parameters and make 2-d histograms of them
for region in regions:
    for measure in ('AIC', 'BIC'):
        for parameter in common_parameters:
            for iwave1, wave1 in enumerate(waves):
                for iwave2 in range(iwave1+1, len(waves)):
                    wave2 = waves[iwave2]