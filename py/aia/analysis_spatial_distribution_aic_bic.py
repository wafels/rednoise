#
# Analysis - AIC and BIC.  Plot the spatial distributions of
# the AIC and the BIC
#
import os
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

import sunpy.map
from copy import deepcopy

import analysis_get_data
import study_details as sd

# Wavelengths we want to analyze
waves = ['171']

# Regions we are interested in
regions = ['sunspot', 'moss', 'quiet Sun', 'loop footpoints']
regions = ['most_of_fov']

# Apodization windows
windows = ['hanning']

# Model results to examine
model_names = ('Power law + Constant + Lognormal', 'Power law + Constant')

# Load in all the data
storage = analysis_get_data.get_all_data(waves=waves, regions=regions)

# Get the sunspot outline
sunspot_outline = analysis_get_data.sunspot_outline()

# Plot spatial distributions of the AIC and BIC.  The AIC and BIC for each
# model are subtracted, and the model with the lowest AIC or BIC is preferred.
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

        # Region submap
        region_submap = analysis_get_data.get_region_submap(output, region_id)

        for measure in ("AIC", "BIC"):
            # Get the information for each model
            this0 = storage[wave][region][model_names[0]]
            this1 = storage[wave][region][model_names[1]]
            # Where are the good fits
            good_fit0 = this0.good_fits()
            good_fit1 = this1.good_fits()
            good_fit_both = np.logical_not(good_fit0) * np.logical_not(good_fit1)

            # Difference in the information criteria
            measure_difference = this0.as_array(measure) - this1.as_array(measure)

            for include_mask in (True, False):

                if include_mask:
                    # Make a masked array
                    map_data = ma.array(measure_difference, mask=np.logical_not(good_fit_both))
                    mask_status = 'with_mask'
                else:
                    map_data = measure_difference
                    mask_status = 'no_mask'

                # Make a SunPy map for nice spatially aware plotting.
                my_map = sunpy.map.Map(map_data, deepcopy(region_submap.meta))
                my_map = analysis_get_data.hsr2015_map(my_map)
                model_name_0 = analysis_get_data.hsr2015_model_name(model_names[0])
                model_name_1 = analysis_get_data.hsr2015_model_name(model_names[1])

                # Make a spatial distribution map of the difference in the
                # information criterion.
                plt.close('all')
                # Normalize the color table so that zero is in the middle
                map_data_abs_max = np.max([np.abs(np.max(map_data)), np.abs(np.min(map_data))])
                norm = colors.Normalize(vmin=-map_data_abs_max, vmax=map_data_abs_max, clip=False)

                # Set up the palette we will use
                palette = cm.seismic
                # Bad values are those that are masked out
                palette.set_bad('y', 1.0)
                # Begin the plot
                fig, ax = plt.subplots()
                # Plot the map
                ret = my_map.plot(cmap=palette, axes=ax, interpolation='none',
                                  norm=norm)
                if region == 'sunspot' or region == 'most_of_fov':
                    ax.add_collection(analysis_get_data.rotate_sunspot_outline(sunspot_outline[0], sunspot_outline[1], my_map.date))

                cbar = fig.colorbar(ret, extend='both', orientation='horizontal',
                                    shrink=0.8, label='$%s_{%s}$-$%s_{%s}$' % (measure, model_name_0, measure, model_name_1))
                # Fit everything in.
                ax.autoscale_view()

                # Dump to file
                filepath = os.path.join(image, 'spatial_distrib.' + region_id + '.%s.%s.png' % (measure, mask_status))
                print('Saving to ' + filepath)
                plt.savefig(filepath)
