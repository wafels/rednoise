#
# Analysis - distributions.  Load in all the data and make some population
# and spatial distributions
#
import os
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import analysis_get_data
import study_details as sd
from analysis_details import rchi2limitcolor, limits, get_mask_info

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

# Number of bins
bins = 100

# Plot cross-correlations of
plot_type = 'cc.within'
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
            parameters = this.model.parameters

            npar = len(parameters)
            for i in range(0, npar):
                # First parameter name
                p1_name = parameters[i]
                # First parameter, label for the plot
                p1_label = this.model.labels[i]
                # First parameter, data
                p1 = this.as_array(p1_name)
                # First parameter, good fits mask
                p1_mask = this.good_fits()
                # Mask out extreme values
                p1_mask[np.where(p1 < limits[p1_name][0])] = 1
                p1_mask[np.where(p1 > limits[p1_name][1])] = 1

                for j in range(i+1, npar):
                    # Second parameter name
                    p2_name = parameters[j]
                    # Second parameter, label for the plot
                    p2_label = this.model.labels[j]
                    # First parameter, data
                    p2 = this.as_array(p2_name)
                    # Second parameter, good fits mask
                    p2_mask = this.good_fits()
                    # Mask out extreme values
                    p2_mask[np.where(p2 < limits[p2_name][0])] = 1
                    p2_mask[np.where(p2 > limits[p2_name][1])] = 1

                    # Final mask for cross-correlation
                    final_mask = np.logical_not(np.logical_not(p1_mask) * np.logical_not(p2_mask))
                    title = wave + "-" + region + get_mask_info(final_mask)

                    # Get the data using the final mask
                    p1 = np.ma.array(this.as_array(p1_name), mask=final_mask).compressed()
                    p2 = np.ma.array(this.as_array(p2_name), mask=final_mask).compressed()

                    # Cross correlation statistics
                    r = [spearmanr(p1, p2), pearsonr(p1, p2)]

                    # Form the rank correlation string
                    rstring = 'spr=%1.2f_pea=%1.2f' % (r[0][0], r[1][0])

                    # Identifier of the plot
                    plotident = rstring + '.' + wave + '.' + region + '.' + p1_name + '.' + p2_name

                    # Make a scatter plot
                    plt.close('all')
                    plt.title(title)
                    plt.xlabel(p1_label)
                    plt.ylabel(p2_label)
                    plt.scatter(p1, p2)
                    x0 = plt.xlim()[0]
                    ylim = plt.ylim()
                    y0 = ylim[0] + 0.3 * (ylim[1] - ylim[0])
                    y1 = ylim[0] + 0.6 * (ylim[1] - ylim[0])
                    plt.text(x0, y0, 'Pearson=%f' % r[0][0], bbox=dict(facecolor=rchi2limitcolor[1], alpha=0.5))
                    plt.text(x0, y1, 'Spearman=%f' % r[1][0], bbox=dict(facecolor=rchi2limitcolor[0], alpha=0.5))
                    ofilename = this_model + '.' + plot_type + '.scatter.' + plotident + '.png'
                    plt.tight_layout()
                    plt.savefig(os.path.join(image, ofilename))

                    # Make a 2d histogram
                    plt.close('all')
                    plt.title(title)
                    plt.xlabel(p1_label)
                    plt.ylabel(p2_label)
                    plt.hist2d(p1, p2, bins=bins)
                    x0 = plt.xlim()[0]
                    ylim = plt.ylim()
                    y0 = ylim[0] + 0.3 * (ylim[1] - ylim[0])
                    y1 = ylim[0] + 0.6 * (ylim[1] - ylim[0])
                    plt.text(x0, y0, 'Pearson=%f' % r[0][0], bbox=dict(facecolor=rchi2limitcolor[1], alpha=0.5))
                    plt.text(x0, y1, 'Spearman=%f' % r[1][0], bbox=dict(facecolor=rchi2limitcolor[0], alpha=0.5))
                    ofilename = this_model + '.' + plot_type + '.hist2d.' + plotident + '.png'
                    plt.tight_layout()
                    plt.savefig(os.path.join(image, ofilename))
