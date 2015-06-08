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
import copy
import numpy as np
import matplotlib.pyplot as plt
from astroML.plotting import hist
import analysis_get_data
import study_details as sd
from analysis_details import convert_to_period, summary_statistics, get_mode

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
hloc = (100, 'blocks', 'scott', 'knuth', 'freedman')

# Period limit
period_limit = 3000.0
ratio_limit = 5.0

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
            mask[np.where(p1 > period_limit)] = 1
            # Masked arrays
            p1 = np.ma.array(p1, mask=mask).compressed()
            # Summary stats
            ss = summary_statistics(p1)

            # Title of the plot
            title = wave + '-' + region + '(#pixels=%i, used=%3.1f%%)' % (n_mask, 100 * n_mask/ np.float64(mask.size))

            # Identifier of the plot
            plotident = wave + '.' + region + '.' + 'time-scale'

            # For what it is worth, plot the same data using all the bin
            # choices.
            plt.close('all')
            plt.figure(1, figsize=(10,10))
            for ibinning, binning in enumerate(hloc):
                plt.subplot(len(hloc), 1, ibinning+1)
                h_info = hist(p1, bins=binning)
                mode = get_mode(h_info)
                plt.axvline(ss['mean'], color='r', label='mean=%f' % ss['mean'])
                plt.axvline(mode[1][0], color='g', label='%f<mode<%f' % (mode[1][0], mode[1][1]))
                plt.axvline(mode[1][1], color='g')
                plt.axvline(300, color='k', label='5 minutes', linestyle="-")
                plt.axvline(180, color='k', label='3 minutes', linestyle=":")
                plt.xlabel('Time-scale of location')
                plt.title(str(binning) + ' : ' + title)
                plt.legend(framealpha=0.5, fontsize=8)

            plt.tight_layout()
            ofilename = this_model + '.hist.' + plotident + '.png'
            plt.savefig(os.path.join(image, ofilename))

            #
            # Ratio of the peak of the lognormal to the power law
            #
            fn = this.fn
            ratio_max = np.zeros((this.ny, this.nx))
            ratio_max_f = np.zeros_like(ratio_max)
            for i in range(0, this.nx):
                for j in range(0, this.ny):
                    estimate = this.result[j][i][1]['x']
                    power_law = this.model.power_per_component(estimate, fn)[0]
                    lognormal = this.model.power_per_component(estimate, fn)[2]
                    ratio = lognormal/power_law
                    ratio_max[j, i] = np.log10(np.max(ratio))
                    ratio_max_f[j, i] = 1.0 / this.f[np.argmax(ratio)]
            new_mask = copy.deepcopy(mask)
            too_small = np.where(ratio_max < -ratio_limit)
            too_big = np.where(ratio_max > ratio_limit)
            new_mask[too_small] = 1
            new_mask[too_big] = 1
            ratio_max = np.ma.array(ratio_max, mask=new_mask).compressed()
            # Summary stats
            ss = summary_statistics(ratio_max)

            # Title of the plot
            title = wave + '-' + region + '(#pixels=%i, used=%3.1f%%)' % (n_mask, 100 * n_mask/ np.float64(mask.size))

            # Identifier of the plot
            plotident = wave + '.' + region + '.' + 'ratio(maximum)'

            # Ratio maximum
            plt.close('all')
            plt.figure(1, figsize=(10, 10))
            for ibinning, binning in enumerate(hloc):
                plt.subplot(len(hloc), 1, ibinning+1)
                h_info = hist(ratio_max, bins=binning)
                mode = get_mode(h_info)
                plt.axvline(ss['mean'], color='r', label='mean=%f' % ss['mean'])
                plt.axvline(mode[1][0], color='g', label='%f<mode<%f' % (mode[1][0], mode[1][1]))
                plt.axvline(mode[1][1], color='g')
                plt.xlabel('$log_{10}\max$(lognormal / power law)')
                plt.title(str(binning) + ' : ' + title)
                plt.legend(framealpha=0.5, fontsize=8)
                plt.xlim(-ratio_limit, ratio_limit)

            plt.tight_layout()
            ofilename = this_model + '.hist.' + plotident + '.png'
            plt.savefig(os.path.join(image, ofilename))

            # location of the ratio maximum
            # Title of the plot
            title = wave + '-' + region + '(#pixels=%i, used=%3.1f%%)' % (n_mask, 100 * n_mask/ np.float64(mask.size))
            # Identifier of the plot
            plotident = wave + '.' + region + '.' + 'argmax(ratio)'
            plt.close('all')
            plt.figure(1, figsize=(10, 10))
            ratio_max_f = np.ma.array(ratio_max_f, mask=new_mask).compressed()
            # Summary stats
            ss = summary_statistics(ratio_max_f)
            for ibinning, binning in enumerate(hloc):
                plt.subplot(len(hloc), 1, ibinning+1)
                h_info = hist(ratio_max_f, bins=binning)
                mode = get_mode(h_info)
                plt.axvline(ss['mean'], color='r', label='mean=%f' % ss['mean'])
                plt.axvline(mode[1][0], color='g', label='%f<mode<%f' % (mode[1][0], mode[1][1]))
                plt.axvline(mode[1][1], color='g')
                plt.xlabel('argmax(lognormal / power law) [seconds]')
                plt.title(str(binning) + ' : ' + title)
                plt.legend(framealpha=0.5, fontsize=8)

            plt.tight_layout()
            ofilename = this_model + '.hist.' + plotident + '.png'
            plt.savefig(os.path.join(image, ofilename))
