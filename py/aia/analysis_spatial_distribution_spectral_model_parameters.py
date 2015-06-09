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
import study_details as sd
from analysis_details import limits, get_mask_info

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



        for model_name in model_names:
            # Get the data for this model
            this = storage[wave][region][model_name]

            # Parameters
            parameters = this.model.parameters

            for parameter in parameters:
                # Where are the good fits
                mask = this.good_fits()

                # Data
                p1 = this.as_array(parameter)

                # Apply the limit masks
                #mask[np.where(p1 < limits[parameter][0])] = 1
                #mask[np.where(p1 > limits[parameter][1])] = 1

                # Create the masked numpy array
                map_data = ma.array(p1, mask=mask)
                # Make a SunPy map for nice spatially aware plotting.
                my_map = analysis_get_data.make_map(output, region_id, map_data)

                # Make a spatial distribution map spectral model parameter
                plt.close('all')
                # Normalize the color table
                norm = colors.Normalize(clip=False, vmin=limits[parameter][0], vmax=limits[parameter][1])

                # Set up the palette we will use
                palette = cm.Greys
                # Bad values are those that are masked out
                palette.set_bad('yellow', 1.0)
                palette.set_under('green', 1.0)
                palette.set_over('red', 1.0)
                # Begin the plot
                fig, ax = plt.subplots()
                # Plot the map
                ret = my_map.plot(cmap=palette, axes=ax, interpolation='none',
                                  norm=norm)
                if region == 'sunspot':
                    ax.add_collection(analysis_get_data.rotate_sunspot_outline(sunspot_outline[0], sunspot_outline[1], my_map.date))

                label_index = this.model.parameters.index(parameter)
                cbar = fig.colorbar(ret, extend='both', orientation='horizontal',
                                    shrink=0.8, label=this.model.labels[label_index])
                # Fit everything in.
                ax.autoscale_view()

                # Dump to file
                filepath = os.path.join(image, '%s.spatial_distrib.' % (model_name,)  + region_id + '.%s.png' % parameter)
                print('Saving to ' + filepath)
                plt.savefig(filepath)
