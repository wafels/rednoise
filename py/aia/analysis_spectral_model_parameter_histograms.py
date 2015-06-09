#
# Analysis - 1-d histograms of parameter values
#
import os
import numpy as np
import matplotlib.pyplot as plt
from astroML.plotting import hist
import analysis_get_data
import study_details as sd
from analysis_details import summary_statistics, get_mode, limits, get_mask_info

# Wavelengths we want to analyze
waves = ['171', '193']

# Regions we are interested in
regions = ['sunspot', 'moss', 'quiet Sun', 'loop footpoints']

# Apodization windows
windows = ['hanning']

# Model results to examine
model_names = ('Power law + Constant + Lognormal','Power law + Constant')

# Load in all the data
storage = analysis_get_data.get_all_data(waves=waves)

# Number of bins
hloc = (100, 'blocks', 'scott', 'knuth', 'freedman')

# Line width
linewidth=3

for wave in waves:
    for region in regions:

        # branch location
        b = [sd.corename, sd.sunlocation, sd.fits_level, wave, region]

        # Region identifier name
        region_id = sd.datalocationtools.ident_creator(b)

        # Output location
        output = sd.datalocationtools.save_location_calculator(sd.roots, b)["pickle"]
        image = sd.datalocationtools.save_location_calculator(sd.roots, b)["image"]

        for this_model in model_names:
            this = storage[wave][region][this_model]

            for p1_name in this.model.parameters:
                p1 = this.as_array(p1_name)
                p1_index = this.model.parameters.index(p1_name)
                label1 = this.model.labels[p1_index]

                # Mask where the fit is good.
                mask = this.good_fits()

                # Apply the limit masks
                mask[np.where(p1 < limits[p1_name][0])] = 1
                mask[np.where(p1 > limits[p1_name][1])] = 1

                # Masked arrays
                p1 = np.ma.array(p1, mask=mask).compressed()

                # Summary stats
                ss = summary_statistics(p1)

                # Title of the plot
                title = wave + "-" + region + get_mask_info(mask)

                # Identifier of the plot
                plotident = wave + '.' + region + '.' + p1_name

                # For what it is worth, plot the same data using all the bin
                # choices.
                plt.close('all')
                plt.figure(1, figsize=(10, 10))
                for ibinning, binning in enumerate(hloc):
                    plt.subplot(len(hloc), 1, ibinning+1)
                    h_info = hist(p1, bins=binning)
                    mode = get_mode(h_info)
                    plt.axvline(ss['mean'], color='r', label='mean=%f' % ss['mean'], linewidth=linewidth)
                    plt.axvline(mode[1][0], color='g', label='%f<mode<%f' % (mode[1][0], mode[1][1]), linewidth=linewidth)
                    plt.axvline(mode[1][1], color='g', linewidth=linewidth)
                    plt.xlabel(label1)
                    plt.title(str(binning) + ' : ' + title)
                    plt.legend(framealpha=0.5, fontsize=8)
                    plt.xlim(limits[p1_name])

                plt.tight_layout()
                ofilename = this_model + '.hist.' + plotident + '.png'
                plt.savefig(os.path.join(image, ofilename))
