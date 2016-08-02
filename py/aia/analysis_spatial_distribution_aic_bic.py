#
# Analysis - AIC and BIC.  Plot the spatial distributions of
# the AIC and the BIC
#
import os
import pickle
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

import sunpy.map
from copy import deepcopy

import analysis_get_data
import analysis_explore
import details_study as ds
import details_analysis as da

# Wavelengths we want to analyze
waves = ['171']

# Regions we are interested in
regions = ['six_euv']

# Apodization windows
windows = ['hanning']

# Model results to examine
model_names = ('Power Law + Constant + Lognormal', 'Power Law + Constant')
model_paper_names = ('2', '1')

# Use the combined good fits and parameter limits
include_masks = (True, False)

# Show the percentages of each model
show_percentage = False

# Limits we are applying
limit_type = 'standard'

# Details of the analysis
limits = da.limits[limit_type]
ic_types = da.ic_details.keys()

# Load in all the data
storage = analysis_get_data.get_all_data(waves=waves,
                                         regions=regions,
                                         model_names=model_names,
                                         spectral_model='.rnspectralmodels3')
mdefine = analysis_explore.MaskDefine(storage, limits)

# Get the sunspot outline
sunspot_outline = analysis_get_data.sunspot_outline()

# Storage for the filepaths
filepaths_root = os.path.join(ds.dataroot, 'pickle')
filepaths = []

# Plot spatial distributions of the AIC and BIC.  The AIC and BIC for each
# model are subtracted, and the model with the lowest AIC or BIC is preferred.
for wave in waves:
    for region in regions:

        # branch location
        b = [ds.corename, ds.sunlocation, ds.fits_level, wave, region]

        # Region identifier name
        region_id = ds.datalocationtools.ident_creator(b)

        # Output location
        output = ds.datalocationtools.save_location_calculator(ds.roots, b)["pickle"]
        image = ds.datalocationtools.save_location_calculator(ds.roots, b)["image"]

        # Output filename
        ofilename = os.path.join(output, region_id + '.datacube')

        # Region submap
        region_submap = analysis_get_data.get_region_submap(output, region_id)['reference region']

        # Go through each measure
        for measure in ic_types:

            # Go through each limit
            for this_ic_limit in da.ic_details[measure]:

                # Get the information for each model
                this0 = storage[wave][region][model_names[0]]
                this1 = storage[wave][region][model_names[1]]

                # What masks do we require here?  We want an assessment of the BIC.  So all
                # we really need here is to know if the fit algorithm marked success or not.
                mask0 = np.logical_not(this0.as_array("success"))
                mask1 = np.logical_not(this1.as_array("success"))
                #mask0 = mdefine.combined_good_fit_parameter_limit[wave][region][model_names[0]]
                #mask1 = mdefine.combined_good_fit_parameter_limit[wave][region][model_names[1]]
                mask = np.logical_or(mask0, mask1)

                # Fraction of fits that are unsuccessful
                unsuccessful_fit_fraction = np.sum(mask)/mask.size
                print('Fraction unsuccessful = {:n}'.format(unsuccessful_fit_fraction))

                # Difference in the information criteria
                measure_difference = this0.as_array(measure) - this1.as_array(measure)

                # Preference for model 0
                model0_where = np.where((measure_difference <= -this_ic_limit) & np.logical_not(mask))
                mask[model0_where] = False
                n_model0 = len(model0_where[0])
                print(measure, ' Model 0 number preferred ', n_model0)

                # Preference for model 1
                model1_where = np.where((measure_difference >= this_ic_limit) & np.logical_not(mask))
                mask[model1_where] = False
                n_model1 = len(model1_where[0])
                print(measure, ' Model 1 number preferred ', n_model1)

                for include_mask in include_masks:

                    if include_mask:
                        # Make a masked array
                        map_data = ma.array(measure_difference, mask=mask)
                        mask_status = 'with_mask'
                    else:
                        map_data = measure_difference
                        mask_status = 'no_mask'

                    # Make a SunPy map for nice spatially aware plotting.
                    my_map = sunpy.map.Map(map_data, deepcopy(region_submap.meta))
                    #my_map = analysis_get_data.hsr2015_map(my_map)
                    #model_name_0 = analysis_get_data.hsr2015_model_name(model_names[0])
                    #model_name_1 = analysis_get_data.hsr2015_model_name(model_names[1])
                    model_name_0 = model_paper_names[0]
                    model_name_1 = model_paper_names[1]

                    fraction_model0 = n_model0 / (1.0 * mask.size)
                    model_name_0_ltx = '$M_' + model_name_0 + '$'
                    percent_model0_string = '%s: %.1f%%' % (model_name_0_ltx, 100.0 * fraction_model0)
                    print('Fraction model 0 = {:n}'.format(fraction_model0))

                    fraction_model1 = n_model1 / (1.0 * mask.size)
                    model_name_1_ltx = '$M_' + model_name_1 + '$'
                    percent_model1_string = '%s: %.1f%%' % (model_name_1_ltx, 100.0 * fraction_model1)
                    print('Fraction model 1 = {:n}'.format(fraction_model1))

                    remainder = (mask.size - (n_model0 + n_model1)) / (1.0 * mask.size)
                    print('Fraction remainder = {:n}'.format(remainder))

                    # Make a spatial distribution map of the difference in the
                    # information criterion.
                    plt.close('all')
                    # Normalize the color table so that zero is in the middle
                    map_data_abs_max = np.max([np.abs(np.max(map_data)), np.abs(np.min(map_data))])
                    # for hsr 2015
                    if measure == 'AIC':
                        map_data_abs_max = 905.0
                    else:
                        map_data_abs_max = this_ic_limit
                    norm = colors.Normalize(vmin=-map_data_abs_max, vmax=map_data_abs_max, clip=False)
                    print(measure, map_data_abs_max)

                    # Set up the palette we will use
                    palette = cm.viridis
                    palette = cm.plasma
                    # Bad values are those that are masked out
                    palette.set_bad('black', 1.0)
                    palette.set_under('blue', 1.0)
                    palette.set_over('cyan', 1.0)
                    # Begin the plot
                    fig, ax = plt.subplots()
                    # Plot the map
                    ret = my_map.plot(cmap=palette, axes=ax, interpolation='none',
                                      norm=norm)
                    label = '$\Delta %s$ = $%s_{%s}$-$%s_{%s}$' % (measure, measure, model_name_0, measure, model_name_1)
                    ret.axes.set_title("%s\n%s " % ('AIA ' + wave + ' Angstrom', label))
                    # Get the sunspot outline
                    sunspot_outline = analysis_get_data.sunspot_outline()
                    _polygon, collection = analysis_get_data.rotate_sunspot_outline(sunspot_outline[0], sunspot_outline[1], region_submap.date, edgecolors=['w'])
                    ax.add_collection(collection)

                    cbar = fig.colorbar(ret, extend='both', orientation='vertical',
                                        shrink=0.8, label=label)

                    if show_percentage:
                        ax.text(-240, -50, percent_model1_string, color='black',
                                bbox=dict(facecolor='white', edgecolor='black', alpha=0.9))
                        ax.text(-240, -75, percent_model0_string, color='black',
                                bbox=dict(facecolor='white', edgecolor='black', alpha=0.9))

                    # Fit everything in.
                    ax.autoscale_view()

                    # Dump image to file
                    save_filename = 'spatial_distrib.' + region_id + '.%s.%s.%s.eps' % (measure, str(this_ic_limit), mask_status)
                    filepath = os.path.join(image, save_filename)
                    print('Saving to ' + filepath)
                    plt.savefig(filepath)
                    plt.close('all')

                    # Dump the data used to a pickle file
                    filepath = os.path.join(output, save_filename)
                    f = open(filepath, 'wb')
                    pickle.dump(my_map, f)
                    pickle.dump(unsuccessful_fit_fraction, f)
                    pickle.dump(fraction_model0, f)
                    pickle.dump(fraction_model1, f)
                    pickle.dump(remainder, f)
                    f.close()

                    # Store the location and file name
                    filepaths.append(filepath)

# Save the location and file name data
filepaths_filepath = os.path.join(filepaths_root, "{:s}.filepaths.pkl".format(__name__))
f = open(filepath, 'wb')
pickle.dump(filepaths, f)
f.close()
