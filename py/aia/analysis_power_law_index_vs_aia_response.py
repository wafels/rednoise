#
# Analysis - power law index versus aia response
#
import os
import numpy as np
from scipy.io import readsav
import matplotlib.pyplot as plt
#from astroML.plotting import hist
import analysis_get_data
import details_study as ds
import details_analysis as da
import details_plots as dp
import analysis_explore



# Paper 2: Wavelengths and regions
waves = ['94', '335', '131', '171', '193', '211']
regions = ['six_euv']

# Paper 3: Wavelengths and regions
#waves = ['171']
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
storage = analysis_get_data.get_all_data(waves=waves, regions=regions)

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
fontsize = dp.fontsize

#
# Read in the AIA responses
#
aia_resp = readsav(os.path.expanduser('~/Desktop/simple_aia.sav'))

#
# All the summary stats
#
ss_all = {}


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
                output = ds.datalocationtools.save_location_calculator(ds.roots, b)["image"]

                for this_model in ('Power Law + Constant',):
                    # Get the data
                    this = storage[wave][region][this_model]

                    # Get the combined mask
                    mask1 = mdefine.combined_good_fit_parameter_limit[wave][region][this_model]

                    # Get the parameters
                    parameters = ['power law index', ]

                    # Get the labels
                    labels = ['n', ]

                    for p1_name in parameters:
                        p1 = this.as_array(p1_name)
                        p1_index = parameters.index(p1_name)
                        label1 = labels[p1_index]
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
                            ss_all[wave] = da.summary_statistics(pm1)

        max_response_logt = []
        power_law_index_mode = []
        yerr = []
        for wave in waves:
            logt = aia_resp['a%st' % wave]
            r = aia_resp['a%sr' % wave]
            max_response_index = np.argmax(r)
            max_response_logt.append(logt[max_response_index])
            power_law_index_mode.append(ss_all[wave]['mode'].value)
            yerr.append(ss_all[wave]['std'].value)

        #argsort = np.argsort(max_response_logt)

        # Make the plot
        plt.close('all')
        plt.xlabel(r'log$_{10}$(temperature of max. response)', fontsize=fontsize)
        plt.ylabel('mode of power law index', fontsize=fontsize)
        #
        # Every point will have to have a label attached to it and an error bar
        #
        plt.errorbar(max_response_logt, power_law_index_mode, yerr=yerr, fmt='o')
        for label, x, y in zip(waves,
                               max_response_logt,
                               power_law_index_mode):
            plt.annotate(
                label,
                xy=(x, y), xytext=(-20, 20),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
                arrowprops = dict(arrowstyle='->', connectionstyle = 'arc3,rad=0'))
        plt.tight_layout()
        ofilename = this_model + '.aia_response_vs_plaw_index.png'
        plt.savefig(os.path.join(image, ofilename))
