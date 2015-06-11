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
from analysis_details import limits, get_mask_info, get_image_model_location, get_ic_location

# Wavelengths we want to analyze
waves = ['171', '193']

# Regions we are interested in
regions = ['moss', 'sunspot', 'quiet Sun', 'loop footpoints']

# Apodization windows
windows = ['hanning']

# Model results to examine
model_names = ('Power law + Constant + Lognormal', 'Power law + Constant')

# IC
ic_types = ('none', 'AIC', 'BIC', 'both')

# Load in all the data
storage = analysis_get_data.get_all_data(waves=waves)

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
