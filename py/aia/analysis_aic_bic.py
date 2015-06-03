#
# Analysis - AIC and BIC.  Plot the spatial distributions of
# the AIC and the BIC
#
import os
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import analysis_get_data
import study_details as sd

# Wavelengths we want to analyze
waves = ['171', '193']

# Regions we are interested in
regions = ['sunspot', 'moss', 'quiet Sun', 'loop footpoints']

# Apodization windows
windows = ['hanning']

# Model results to examine
model_names = ('Power law + Constant + Lognormal', 'Power law + Constant')

# Load in all the data
storage = analysis_get_data.get_all_data(waves=waves)

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

        for measure in ("AIC", "BIC"):
            # Get the information for each model
            this0 = storage[wave][region][model_names[0]]
            this1 = storage[wave][region][model_names[1]]
            # Where are the good fits
            good_fit0 = this0.good_fits()
            good_fit1 = this1.good_fits()
            good_fit_both = good_fit0 * good_fit1

            # Difference in the information criteria
            measure_difference = this0.as_array(measure) - this1.as_array(measure)

            # Make a masked array
            map_data = ma.array(measure_difference, mask=good_fit_both)

            # Make a SunPy map for nice spatially aware plotting.
            my_map = analysis_get_data.make_map(output, region_id, map_data)

            # Make a spatial distribution map that also shows where the bad
            # fits are
            plt.close('all')
            # Set up the palette we will use
            palette = cm.gnuplot2
            # Bad values are those that are masked out
            palette.set_bad('0.75')

            # Begin the plot
            fig, ax = plt.subplots()
            # Plot the map
            ret = my_map.plot(cmap=palette, axes=ax, interpolation='none')

            cbar = fig.colorbar(ret, extend='both', orientation='horizontal',
                                shrink=0.8, label='$%s_{%s}$-$%s_{%s}$' % (measure, model_names[0], measure, model_names[1]))

            # Fit everything in.
            ax.autoscale_view()

            # Dump to file
            filepath = os.path.join(image, 'spatial_distrib.' + region_id + '.%s.png' % measure)
            print('Saving to ' + filepath)
            plt.savefig(filepath)
