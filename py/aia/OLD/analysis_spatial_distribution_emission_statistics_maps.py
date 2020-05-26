#
# Analysis - Plot the average emission over the region.  This can be compared
# to the analysis results
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
waves = ['131', '171', '193', '211']

# Regions we are interested in
# regions = ['sunspot', 'moss', 'quiet Sun', 'loop footpoints']
# regions = ['most_of_fov']
regions = ['four_wavebands']

# Apodization windows
windows = ['hanning']

# Model results to examine
model_names = ('Power Law + Constant + Lognormal', 'Power Law + Constant')

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
bins = 100


# Load in all the data
storage = analysis_get_data.get_all_data(waves=waves,
                                         regions=regions,
                                         model_names=model_names)
mdefine = analysis_explore.MaskDefine(storage, limits)


# Get the sunspot outline
sunspot_outline = analysis_get_data.sunspot_outline()


# Plot cross-correlations across different AIA channels
plot_type = 'spatial.average_emission'

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
        region_maps = analysis_get_data.get_region_submap(output, region_id)

        # Map types
        map_types = region_maps.keys()

        for map_type in map_types:
            this_submap = region_maps[map_type]

            image = dp.get_image_model_location(ds.roots, b, [])

            subtitle_filename = dp.concat_string([region, map_type], sep='.')

            # Plot identity
            plot_identity_filename = dp.concat_string([wave,])

            # Get the sunspot
            sunspot_collection = analysis_get_data.rotate_sunspot_outline(sunspot_outline[0], sunspot_outline[1], this_submap.date)

            # Make a spatial distribution map spectral model parameter
            plt.close('all')

            # Begin the plot
            fig, ax = plt.subplots()
            # Plot the map
            ret = this_submap.plot(axes=ax, interpolation='none')
            ret.axes.set_title('%s\n%s' % (map_type, this_submap.name))
            ax.add_collection(sunspot_collection)
            # Fit everything in.
            ax.autoscale_view()

            # Dump to file
            final_filename = dp.concat_string([plot_type,
                                               plot_identity_filename,
                                               subtitle_filename]) + '.png'
            filepath = os.path.join(image, final_filename)
            print('Saving to ' + filepath)
            plt.savefig(filepath)
