#
# Analysis - 1-d histograms of parameter values
#
import os
import numpy as np
import matplotlib.pyplot as plt
from astroML.plotting import hist
import analysis_get_data
import study_details as sd
from analysis_details import summary_statistics, get_mode, limits, get_mask_info, get_ic_location, get_image_model_location
import analysis_explore

# Wavelengths we want to analyze
waves = ['171']#, '193']

# Regions we are interested in
regions = ['sunspot', 'moss', 'quiet Sun', 'loop footpoints']
regions = ['most_of_fov']
# Apodization windows
windows = ['hanning']

# IC
ic_types = ('BIC')
ic_limit = 6.0

# Load in all the data
storage = analysis_get_data.get_all_data(waves=waves)

# Define the masks
mdefine = analysis_explore.MaskDefine(storage)
available_models = mdefine.available_models

# Number of bins
hloc = (100, 'blocks', 'scott', 'knuth', 'freedman')

# Line width
linewidth = 3

for wave in waves:
    for region in regions:

        # branch location
        b = [sd.corename, sd.sunlocation, sd.fits_level, wave, region]

        # Region identifier name
        region_id = sd.datalocationtools.ident_creator(b)

        # Output location
        output = sd.datalocationtools.save_location_calculator(sd.roots, b)["pickle"]

        for this_model in available_models:
            # Get the data
            this = storage[wave][region][this_model]

            # Get the parameter limit masks
            mask_aplm = mdefine.all_parameter_limit_masks[wave][region][this_model]

            # Get the good fit mask
            mask_gfm = mdefine.good_fit_masks[wave][region][this_model]

            for p1_name in this.model.parameters:
                p1 = this.as_array(p1_name)
                p1_index = this.model.parameters.index(p1_name)
                label1 = this.model.labels[p1_index]
                for ic_type in ic_types:

                    model_by_ic = mdefine.which_model_is_preferred(ic_type, ic_limit)

                    mask_ic = mdefine.is_this_model_is_preferred(ic_type, ic_limit, this_model)

                    # Final mask combines where the parameters are all nice,
                    # where a good fit was achieved, and where the IC limit
                    # criterion was satisfied.
                    mask = np.logical_or(np.logical_or(mask_aplm, mask_gfm), mask_ic)

                    # Masked arrays
                    pm1 = np.ma.array(p1, mask=mask).compressed()

                    # Summary stats
                    ss = summary_statistics(pm1)

                    # Identifier of the plot
                    plot_identity = wave + '.' + region + '.' + p1_name + '.' + ic_type

                    # Title of the plot
                    title = plot_identity + get_mask_info(mask)

                    # location of the image
                    image = get_image_model_location(sd.roots, b, [this_model, ic_type])

                    # For what it is worth, plot the same data using all the bin
                    # choices.
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
                        plt.title(str(binning) + ' : %s\n%s' % (title, this_model))
                        plt.legend(framealpha=0.5, fontsize=8)
                        plt.xlim(limits[p1_name])

                    plt.tight_layout()
                    ofilename = this_model + '.' + plot_identity + '.hist.png'
                    plt.savefig(os.path.join(image, ofilename))
