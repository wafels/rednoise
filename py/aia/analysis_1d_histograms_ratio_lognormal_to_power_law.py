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
# (4)
# Distributions of the energy flux from each contribution
#
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from astroML.plotting import hist
import analysis_get_data
import study_details as sd
from analysis_details import convert_to_period, summary_statistics, get_mode, limits, get_mask_info, get_ic_location, get_image_model_location

# Wavelengths we want to analyze
waves = ['171', '193']

# Regions we are interested in
regions = ['sunspot', 'moss', 'quiet Sun', 'loop footpoints']

# Apodization windows
windows = ['hanning']

# Model results to examine
model_names = ('Power law + Constant + Lognormal',)

# Model results to examine
model_comparison_names = ('Power law + Constant + Lognormal', 'Power law + Constant')

# Load in all the data
storage = analysis_get_data.get_all_data(waves=waves)

# Number of bins
hloc = (100, 'blocks', 'scott', 'knuth', 'freedman')

# Period limit
period_limit = limits["period"]
ratio_limit = limits["ratio"]

# IC
ic_types = ('none', 'AIC', 'BIC', 'both')

linewidth=3

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

        for this_model in model_names:
            for ic_type in ic_types:
                this = storage[wave][region][this_model]
                p1_name = 'log10(lognormal position)'
                p1_index = this.model.parameters.index(p1_name)
                label1 = this.model.labels[p1_index]

                # Plot out the time-scale of the location of the lognormal
                # convert to a period
                f_norm = this.f[0]
                p1 = convert_to_period(f_norm, this.as_array(p1_name))

                # Mask where the lognormal fit is good.
                mask = this.good_fits()

                # Apply the limit masks
                # Mask out bad time scales
                mask[np.where(p1 > period_limit[1])] = 1

                # Only consider those pixels where this_model is preferred
                # by the information criteria
                mask[get_ic_location(storage[wave][region],
                                     this_model,
                                     model_comparison_names,
                                     ic_type=ic_type)] = 1

                # Masked arrays
                p1 = np.ma.array(p1, mask=mask).compressed()

                # Summary stats
                ss = summary_statistics(p1)

                # Identifier of the plot
                plot_identity = wave + '.' + region + '.time-scale.' + ic_type

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
                    h_info = hist(p1, bins=binning)
                    mode = get_mode(h_info)
                    plt.axvline(ss['mean'], color='r', label='mean=%f' % ss['mean'], linewidth=linewidth)
                    plt.axvline(mode[1][0], color='g', label='%f<mode<%f' % (mode[1][0], mode[1][1]), linewidth=linewidth)
                    plt.axvline(mode[1][1], color='g', linewidth=linewidth)
                    plt.axvline(300, color='k', label='5 minutes', linestyle="-", linewidth=linewidth)
                    plt.axvline(180, color='k', label='3 minutes', linestyle=":", linewidth=linewidth)
                    plt.xlabel('Time-scale of location')
                    plt.title(str(binning) + ' : %s\n%s' % (title, this_model))
                    plt.legend(framealpha=0.5, fontsize=8)
                    plt.xlim(period_limit)

                plt.tight_layout()
                ofilename = this_model + '.' + plot_identity + '.hist.png'
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

                # Make a mask for these data
                new_mask = copy.deepcopy(mask)
                too_small = np.where(ratio_max < ratio_limit[0])
                too_big = np.where(ratio_max > ratio_limit[1])
                new_mask[too_small] = 1
                new_mask[too_big] = 1
                ratio_max = np.ma.array(ratio_max, mask=new_mask).compressed()
                ratio_max_f = np.ma.array(ratio_max_f, mask=new_mask).compressed()

                # Summary stats
                ss = summary_statistics(ratio_max)

                # Identifier of the plot
                plot_identity = wave + '.' + region + '.ratio(maximum).' + ic_type

                # Title of the plot
                title = plot_identity + get_mask_info(new_mask)

                # Plot
                plt.close('all')
                plt.figure(1, figsize=(10, 10))
                for ibinning, binning in enumerate(hloc):
                    plt.subplot(len(hloc), 1, ibinning+1)
                    h_info = hist(ratio_max, bins=binning)
                    mode = get_mode(h_info)
                    plt.axvline(ss['mean'], color='r', label='mean=%f' % ss['mean'], linewidth=linewidth)
                    plt.axvline(mode[1][0], color='g', label='%f<mode<%f' % (mode[1][0], mode[1][1]), linewidth=linewidth)
                    plt.axvline(mode[1][1], color='g', linewidth=linewidth)
                    plt.xlabel('$log_{10}\max$(lognormal / power law)')
                    plt.title(str(binning) + ' : %s\n%s' % (title, this_model))
                    plt.legend(framealpha=0.5, fontsize=8)
                    plt.xlim(ratio_limit)

                plt.tight_layout()
                ofilename = this_model + '.' + plot_identity + '.hist.png'
                plt.savefig(os.path.join(image, ofilename))

                # location of the ratio maximum
                # Summary stats
                ss = summary_statistics(ratio_max_f)

                # Identifier of the plot
                plot_identity = wave + '.' + region + '.argmax(ratio).' + ic_type

                # Title of the plot
                title = plot_identity + get_mask_info(new_mask)

                # Plot
                plt.close('all')
                plt.figure(1, figsize=(10, 10))
                for ibinning, binning in enumerate(hloc):
                    plt.subplot(len(hloc), 1, ibinning+1)
                    h_info = hist(ratio_max_f, bins=binning)
                    mode = get_mode(h_info)
                    plt.axvline(ss['mean'], color='r', label='mean=%f' % ss['mean'], linewidth=linewidth)
                    plt.axvline(mode[1][0], color='g', label='%f<mode<%f' % (mode[1][0], mode[1][1]), linewidth=linewidth)
                    plt.axvline(mode[1][1], color='g', linewidth=linewidth)
                    plt.axvline(300, color='k', label='5 minutes', linestyle="-", linewidth=linewidth)
                    plt.axvline(180, color='k', label='3 minutes', linestyle=":", linewidth=linewidth)
                    plt.xlabel('argmax(lognormal / power law) [seconds]')
                    plt.title(str(binning) + ' : %s\n%s' % (title, this_model))
                    plt.legend(framealpha=0.5, fontsize=8)
                    plt.xlim(period_limit)

                plt.tight_layout()
                ofilename = this_model + '.' + plot_identity + '.hist.png'
                plt.savefig(os.path.join(image, ofilename))
