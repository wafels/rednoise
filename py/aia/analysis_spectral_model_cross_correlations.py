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
import analysis_details

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

#
rchi2limitcolor = analysis_details.rchi2limitcolor

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
                p1_name = parameters[i]
                label1 = this.model.labels[i]
                mask = this.good_fits()
                n_mask = np.sum(np.logical_not(mask))
                p1 = np.ma.array(this.as_array(p1_name), mask=mask).compressed()

                title = wave + '-' + region + '(#pixels=%i, used=%3.1f%%)' % (n_mask, 100 * n_mask/ np.float64(mask.size))

                # Identifier of the plot
                plotident = wave + '.' + region + '.' + p1_name

                plt.close('all')
                plt.hist(p1, bins=bins)
                plt.xlabel(label1)
                plt.title(title)
                ofilename = this_model + '.hist.' + plotident + '.png'
                plt.tight_layout()
                plt.savefig(os.path.join(image, ofilename))
                print this_model, wave, region, 'mean ', p1_name, np.mean(p1)

                for j in range(i+1, npar):
                    p2_name = parameters[j]
                    label2 = this.model.labels[j]
                    p2 = np.ma.array(this.as_array(p2_name), mask=mask).compressed()

                    # Cross correlation statistics
                    r = [spearmanr(p1, p2), pearsonr(p1, p2)]

                    # Form the rank correlation string
                    rstring = 'spr=%1.2f_pea=%1.2f' % (r[0][0], r[1][0])

                    # Identifier of the plot
                    plotident = rstring + '.' + wave + '.' + region + '.' + p1_name + '.' + p2_name

                    # Make a scatter plot

                    plt.close('all')
                    plt.title(title)
                    plt.xlabel(label1)
                    plt.ylabel(label2)
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
                    plt.xlabel(label1)
                    plt.ylabel(label2)
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
