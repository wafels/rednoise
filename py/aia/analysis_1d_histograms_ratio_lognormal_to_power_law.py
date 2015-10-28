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
import astropy.units as u
import analysis_get_data
import analysis_explore
import study_details as sd
from analysis_details import convert_to_period, summary_statistics, get_mode, limits, get_mask_info, get_ic_location, get_image_model_location

# Wavelengths we want to analyze
waves = ['211']

# Regions we are interested in
regions = ['sunspot', 'quiet Sun']
#regions = ['most_of_fov']
# Apodization windows
windows = ['hanning']

# Model results to examine
model_names = ('Power Law + Constant + Lognormal',)

# Model results to examine
model_comparison_names = ('Power Law + Constant + Lognormal', 'Power Law + Constant')

# Load in all the data
storage = analysis_get_data.get_all_data(waves=waves,
                                         regions=regions,
                                         model_names=model_comparison_names)

zzz = analysis_explore.MaskDefine(storage)

fz = 'mHz'
three_minutes = 180 * u.s
five_minutes = 300 * u.s


# Number of bins
hloc = (100,)# 'scott', 'knuth', 'freedman')

# Period limit
period_limit = limits["period"]
ratio_limit = limits["ratio"]

# IC
ic_types = ('BIC', 'AIC')
ic_limit = 5.0

linewidth = 3

#
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
                parameters = [v.fit_parameter for v in this.model.variables]
                p1_name = 'lognormal position'
                p1_index = parameters.index(p1_name)

                p1 = this.as_array(p1_name).to(fz)

                mask1 = zzz.combined_good_fit_parameter_limit[wave][region][this_model]
                mask2 = zzz.is_this_model_preferred(ic_type, ic_limit, this_model)[wave][region]
                mask = np.logical_or(mask1, mask2)

                # Masked arrays
                p1 = np.ma.array(p1, mask=mask).compressed()

                # Summary stats
                ss = summary_statistics(p1)

                # Identifier of the plot
                plot_identity = wave + '.' + region + '.frequency.' + ic_type + '>%f' % ic_limit

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
                    plt.hist(p1, bins=binning)
                    plt.axvline(ss['mean'].value, color='r', label='mean=%f %s' % (ss['mean'].value, fz), linewidth=linewidth)
                    plt.axvline(ss['mode'].value, color='g', label='mode=%f %s' % (ss['mode'].value, fz), linewidth=linewidth)
                    plt.axvline((1.0/five_minutes).to(fz).value, color='k', label='5 minutes', linestyle="-", linewidth=linewidth)
                    plt.axvline((1.0/three_minutes).to(fz).value, color='k', label='3 minutes', linestyle=":", linewidth=linewidth)
                    plt.xlabel(p1_name + ' (%s)' % fz)
                    plt.title(str(binning) + ' : %s\n%s' % (title, this_model))
                    plt.legend(framealpha=0.5, fontsize=8)

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
                        ratio_max_f[j, i] = this.f[np.argmax(ratio)]

                # Make a mask for these data
                new_mask = copy.deepcopy(mask)
                too_small = np.where(ratio_max < ratio_limit[0])
                too_big = np.where(ratio_max > ratio_limit[1])
                new_mask[too_small] = True
                new_mask[too_big] = True
                ratio_max = np.ma.array(ratio_max, mask=new_mask).compressed()
                ratio_max_f_normalized_frequencies = np.ma.array(ratio_max_f, mask=new_mask)

                # Summary stats
                ss = summary_statistics(ratio_max)

                # Identifier of the plot
                plot_identity = wave + '.' + region + '.ratio(maximum).' + ic_type + '>%f' % ic_limit

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
                    plt.axvline(ss['mode'], color='g', label='mode=%f' % ss['mode'], linewidth=linewidth)
                    plt.xlabel('$log_{10}\max$(lognormal / power law)')
                    plt.title(str(binning) + ' : %s\n%s' % (title, this_model))
                    plt.legend(framealpha=0.5, fontsize=8)
                    plt.xlim(ratio_limit)

                plt.tight_layout()
                ofilename = this_model + '.' + plot_identity + '.hist.png'
                plt.savefig(os.path.join(image, ofilename))

                # location of the ratio maximum
                # Convert the normalized frequencies to the actual frequencies
                rmf_data = (np.ma.getdata(ratio_max_f_normalized_frequencies)* u.Hz).to(fz).value
                rmf_mask = np.ma.getmask(ratio_max_f_normalized_frequencies)
                too_small = np.where(rmf_data < 0.0)
                too_big = np.where(rmf_data > 10.0)
                rmf_mask[too_small] = True
                rmf_mask[too_big] = True

                ratio_max_f = np.ma.array(rmf_data * u.Unit(fz), mask=rmf_mask).compressed()

                # Summary stats
                ss = summary_statistics(ratio_max_f)

                # Identifier of the plot
                plot_identity = wave + '.' + region + '.argmax(ratio).' + ic_type + '>%f' % ic_limit

                # Title of the plot
                title = plot_identity + get_mask_info(new_mask)

                # Plot
                plt.close('all')
                plt.figure(1, figsize=(10, 10))
                for ibinning, binning in enumerate(hloc):
                    plt.subplot(len(hloc), 1, ibinning+1)
                    h_info = hist(ratio_max_f, bins=binning)
                    mode = get_mode(h_info)
                    plt.axvline(ss['mean'].value, color='r', label='mean=%f %s' % (ss['mean'].value, fz), linewidth=linewidth)
                    plt.axvline(ss['mode'].value, color='g', label='mode=%f %s' % (ss['mode'].value, fz), linewidth=linewidth)
                    plt.axvline((1.0/five_minutes).to(fz).value, color='k', label='5 minutes', linestyle="-", linewidth=linewidth)
                    plt.axvline((1.0/three_minutes).to(fz).value, color='k', label='3 minutes', linestyle=":", linewidth=linewidth)
                    plt.xlabel('argmax(lognormal / power law) (%s)' % fz)
                    plt.title(str(binning) + ' : %s\n%s' % (title, this_model))
                    plt.legend(framealpha=0.5, fontsize=8)
                    plt.xlim(0,10)

                plt.tight_layout()
                ofilename = this_model + '.' + plot_identity + '.hist.png'
                plt.savefig(os.path.join(image, ofilename))

                # Find the frequency at which the estimated power law component
                # is equal to the estimated constant background.
