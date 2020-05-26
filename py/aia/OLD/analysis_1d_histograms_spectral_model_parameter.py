#
# Analysis - 1-d histograms of parameter values
#
import os
import numpy as np
import matplotlib as mpl
import details_plots as dp
mpl.rcParams['xtick.labelsize'] = 0.8*dp.fontsize
mpl.rcParams['ytick.labelsize'] = 0.8*dp.fontsize
import matplotlib.pyplot as plt
import analysis_get_data
import details_study as ds
import details_analysis as da
import analysis_explore

# Paper 2: Wavelengths and regions
waves = ['94', '335', '131', '171', '193', '211']
regions = ['six_euv']

# Paper 3: Wavelengths and regions
#waves = ['193']
#regions = ds.regions


# Paper 3: Wavelengths and regions
#waves = ['335']
#regions = ['six_euv']




# Regions we are interested in
#regions = ['sunspot', 'quiet Sun']
#regions = ['most_of_fov']

# Parameter limits
limit_type = 'standard'

#
# Parameter limits and information criteria
#
limits = da.limits[limit_type]
ic_types = da.ic_details.keys()

# Apodization windows
windows = ['hanning']

# Load in all the data
storage = analysis_get_data.get_all_data(waves=waves,
                                         regions=regions,
                                         model_names=('Power Law + Constant', 'Power Law + Constant + Lognormal'),
                                         spectral_model='.rnspectralmodels4')

# Define the masks
mdefine = analysis_explore.MaskDefine(storage, limits)
available_models = mdefine.available_models

#
# Details of the plotting
#
fz = dp.fz
three_minutes = dp.three_minutes
five_minutes = dp.five_minutes
hloc = dp.hloc
linewidth = 3

# Plot spatial distributions of the spectral model parameters.
# Different information criteria
for ic_type in ic_types:

    # Get the IC limit
    ic_limits = da.ic_details[ic_type]
    for ic_limit in ic_limits:
        ic_limit_string = '%s>%f' % (ic_type, ic_limit)

        for wave in waves:
            for region in regions:

                # branch location
                b = [ds.corename, ds.sunlocation, ds.fits_level, wave, region]

                # Region identifier name
                region_id = ds.datalocationtools.ident_creator(b)

                # Output location
                output = ds.datalocationtools.save_location_calculator(ds.roots, b)["pickle"]

                for this_model in available_models:
                    # Get the data
                    this = storage[wave][region][this_model]

                    # Get the combined mask
                    mask1 = mdefine.combined_good_fit_parameter_limit[wave][region][this_model]

                    # Get the parameters
                    parameters = [v.fit_parameter for v in this.model.variables]

                    # Get the labels
                    labels = [v.converted_label for v in this.model.variables]

                    for p1_name in parameters:
                        p1 = this.as_array(p1_name)
                        p1_index = parameters.index(p1_name)
                        label1 = labels[p1_index]
                        if p1_name == 'lognormal position':
                            label1 += ' (Hz)'
                        for ic_type in ic_types:

                            # Find out if this model is preferred
                            mask2 = mdefine.is_this_model_preferred(ic_type, ic_limit, this_model)[wave][region]

                            # Final mask combines where the parameters are all nice,
                            # where a good fit was achieved, and where the IC limit
                            # criterion was satisfied.
                            mask = np.logical_or(mask1, mask2)

                            # Masked arrays
                            pm1 = np.ma.array(p1, mask=mask).compressed()

                            # Summary stats
                            ss = da.summary_statistics(pm1)

                            # Define the mean and mode lines
                            if p1_name in dp.frequency_parameters:
                                mean_label = 'mean=%f' % ss['mean'].value
                                mode_label = 'mode=%f' % ss['mode'].value
                            else:
                                mean_label = 'mean=%f' % ss['mean'].value
                                mode_label = 'mode=%f' % ss['mode'].value

                            lo68_label = 'low 68%={:n}'.format(ss['lo68'].value)
                            hi68_label = 'high 68%={:n}'.format(ss['hi68'].value)

                            # Identifier of the plot
                            plot_identity = dp.concat_string([wave,
                                                              region,
                                                              p1_name,
                                                              ic_limit_string,
                                                              limit_type])

                            # Title of the plot
                            title = '{:s}\n{:s}\n{:s}\n{:s}'.format(label1, this_model, ic_limit_string, dp.get_mask_info_string(mask)[2])
                            # location of the image
                            image = dp.get_image_model_location(ds.roots, b, [this_model, ic_limit_string, limit_type])

                            # For what it is worth, plot the same data using
                            # all the bin choices.
                            plt.close('all')
                            plt.figure(1, figsize=(10, 10))
                            for ibinning, binning in enumerate(hloc):
                                plt.subplot(len(hloc), 1, ibinning+1)
                                plt.hist(pm1.value, bins=100)
                                plt.axvline(ss['mean'].value,
                                            color=dp.mean.color,
                                            label=mean_label,
                                            linewidth=dp.mean.linewidth,
                                            linestyle=dp.mean.linestyle)
                                plt.axvline(ss['mode'].value,
                                            color=dp.mode.color,
                                            label=mode_label,
                                            linewidth=dp.mode.linewidth,
                                            linestyle=dp.mode.linestyle)
                                plt.axvline(ss['lo68'].value,
                                            color=dp.lo68.color,
                                            label=lo68_label,
                                            linewidth=dp.lo68.linewidth,
                                            linestyle=dp.lo68.linestyle)
                                plt.axvline(ss['hi68'].value,
                                            color=dp.hi68.color,
                                            label=hi68_label,
                                            linewidth=dp.hi68.linewidth,
                                            linestyle=dp.hi68.linestyle)
                                if p1_name in dp.frequency_parameters:
                                    plt.axvline((1.0/five_minutes.position).to(fz).value,
                                                color=five_minutes.color,
                                                label=five_minutes.label,
                                                linestyle=five_minutes.linestyle,
                                                linewidth=five_minutes.linewidth)

                                    plt.axvline((1.0/three_minutes.position).to(fz).value,
                                                color=three_minutes.color,
                                                label=three_minutes.label,
                                                linestyle=three_minutes.linestyle,
                                                linewidth=three_minutes.linewidth)

                                plt.xlabel(label1, fontsize=dp.fontsize)
                                plt.ylabel('number', fontsize=dp.fontsize)
                                plt.title(title, fontsize=dp.fontsize)
                                plt.legend(framealpha=0.5, fontsize=0.8*dp.fontsize)
                                plt.xlim(limits[p1_name].value)

                            plt.tight_layout()
                            ofilename = dp.clean_for_overleaf(this_model + '.' + plot_identity + '.hist.png')
                            plt.savefig(os.path.join(image, ofilename))
