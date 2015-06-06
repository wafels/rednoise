#
# Analysis - examine the lognormal fit.
#
# This program creates the following plots
#
# (1)
# Distributions of the position of the lognormal
# (2)
# Distribution of the location of the maximum of the ratio of the lognormal
# contribution to the background power law
# (3)
# Distribution of the maximum of the ratio of the lognormal contribution to the
# background power law
#
import os
import numpy as np
import matplotlib.pyplot as plt
import analysis_get_data
import study_details as sd
from analysis_details import convert_to_period, summary_statistics

# Wavelengths we want to analyze
waves = ['171', '193']

# Regions we are interested in
regions = ['sunspot', 'moss', 'quiet Sun', 'loop footpoints']

# Apodization windows
windows = ['hanning']

# Model results to examine
model_names = ('Power law + Constant + Lognormal',)

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
            p1_name = 'log10(lognormal position)'

            p1_index = this.model.parameters.index(p1_name)
            label1 = this.model.labels[p1_index]
            #
            # Could also mask by the fits which have good lognormal chi-squared
            # and are significantly preferred by both the AIC and the BIC
            #
            mask = this.good_fits()
            n_mask = np.sum(np.logical_not(mask))

            #
            # Plot out the time-scale of the location of the lognormal
            #
            # convert to a period
            f_norm = this.f[0]
            p1 = convert_to_period(f_norm, this.as_array(p1_name))
            # Mask out the much longer time-scales
            mask[np.where(p1 > 3000)] = 1
            # Masked arrays
            p1 = np.ma.array(p1, mask=mask).compressed()
            # Summary stats
            ss = summary_statistics(p1, bins=bins)

            # Title of the plot
            title = wave + '-' + region + '(#pixels=%i, used=%3.1f%%)' % (n_mask, 100 * n_mask/ np.float64(mask.size))

            # Identifier of the plot
            plotident = wave + '.' + region + '.' + 'time-scale'

            plt.close('all')
            plt.hist(p1, bins=bins)
            plt.axvline(ss['mean'], color='r', label='mean=%f' % ss['mean'])
            plt.axvline(ss['mode'], color='g', label='mode=%f' % ss['mode'])
            plt.axvline(300, color='k', label='5 minutes', linestyle="-")
            plt.axvline(180, color='k', label='3 minutes', linestyle=":")
            plt.xlabel('Time-scale of location')
            plt.title(title)
            ofilename = this_model + '.hist.' + plotident + '.png'
            plt.legend(framealpha=0.5, fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(image, ofilename))

            #
            # Ratio of the peak of the lognormal to the power law
            #
            fn = this.model.fn
            ratio_max = np.zeros((this.model.ny, this.model.nx))
            ratio_max_fn = np.zeros_like(ratio_max)
            for i in range(0, this.model.nx):
                for j in range(0, this.model.ny):
                    estimate = this.result[j][i][1]['x']
                    power_law = this.model.power_per_component(estimate, fn)[0]
                    lognormal = this.model.power_per_component(estimate, fn)[2]
                    ratio = lognormal/power_law
                    ratio_max[j, i] = np.max(ratio)
                    ratio_max_fn[j, i] = convert_to_period(f_norm, fn[np.argmax(ratio)])
