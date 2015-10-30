#
# Analysis - 1-d histograms of parameter values
#
import os
import numpy as np
import matplotlib.pyplot as plt
from astroML.plotting import hist
import analysis_get_data
import details_study as ds
import details_analysis as da
import details_plots as dp
import analysis_explore

# Wavelengths we want to analyze
waves = ['211']#, '193']

# Regions we are interested in
regions = ['sunspot', 'quiet Sun']
#regions = ['most_of_fov']
# Apodization windows
windows = ['hanning']

# Load in all the data
storage = analysis_get_data.get_all_data(waves=waves, regions=regions)

# Define the masks
mdefine = analysis_explore.MaskDefine(storage, limits)
available_models = mdefine.available_models

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
                label1 = labels[p1_index] + ' (%s)' % display_unit[p1_name]
                for ic_type in ic_types:

                    # Get the IC limit
                    ic_limit = da.ic_details[ic_type]

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
                        mean = dp.meanline
                        mean.label = 'mean=%f' % ss['mean'].value
                        mode = dp.modeline
                        mode.label = 'mode=%f' % ss['mode'].value
                    else:
                        mean = dp.meanline
                        mean.label = 'mean=%f' % ss['mean'].value
                        mode = dp.modeline
                        mode.label = 'mode=%f' % ss['mode'].value

                    # Identifier of the plot
                    plot_identity = dp.concat_string([wave,
                                                      region,
                                                      p1_name,
                                                      '%s>%f' % (ic_type, ic_limit)])

                    # Title of the plot
                    title = dp.concat_string([plot_identity,
                                              dp.get_mask_info_string(mask)])

                    # location of the image
                    image = dp.get_image_model_location(ds.roots, b, [this_model, ic_type])

                    # For what it is worth, plot the same data using all the bin
                    # choices.
                    plt.close('all')
                    plt.figure(1, figsize=(10, 10))
                    for ibinning, binning in enumerate(hloc):
                        plt.subplot(len(hloc), 1, ibinning+1)
                        h_info = hist(pm1, bins=binning)
                        plt.axvline(ss['mean'].value,
                                    color=mean.color,
                                    label=mean.label,
                                    linewidth=mean.linewidth)
                        plt.axvline(ss['mode'].value,
                                    color=mode.color,
                                    label=mode.label,
                                    linewidth=mode.linewidth)
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

                        plt.xlabel(label1)
                        plt.title(str(binning) + ' : %s\n%s' % (title, this_model))
                        plt.legend(framealpha=0.5, fontsize=8)
                        plt.xlim(limits[p1_name].value)

                    plt.tight_layout()
                    ofilename = this_model + '.' + plot_identity + '.hist.png'
                    plt.savefig(os.path.join(image, ofilename))
