import os
import numpy as np
from scipy.io import readsav
from scipy.stats import anderson_ksamp
import matplotlib.pyplot as plt
import astropy.units as u
import details_study as ds

# This section loads in the VK time lag data

filepaths = {"low": {"bv": "~/Data/ts/bradshaw_viall_2016/crosscor_SteveModel10.sav",
                     "pli": "~/Data/ts/bradshaw_viall_2016/powerlawindices171/powerlawindices_data_low171.npy",
                     "plim": "~/Data/ts/bradshaw_viall_2016/powerlawindices171/powerlawindices_mask_low171.npy"},
             "intermediate": {"bv": "~/Data/ts/bradshaw_viall_2016/crosscor_SteveTrain10.sav",
                              "pli": "~/Data/ts/bradshaw_viall_2016/powerlawindices171/powerlawindices_data_intermediate171.npy",
                              "plim": "~/Data/ts/bradshaw_viall_2016/powerlawindices171/powerlawindices_mask_intermediate171.npy"},
             "high": {"bv": "~/Data/ts/bradshaw_viall_2016/crosscor_SteveIntTrain10.sav",
                      "pli": "~/Data/ts/bradshaw_viall_2016/powerlawindices171/powerlawindices_data_high171.npy",
                      "plim": "~/Data/ts/bradshaw_viall_2016/powerlawindices171/powerlawindices_mask_high171.npy"}
             }

bv_keys = ('peak335', 'peak211', 'peak193', 'peak171', 'peak094', 'max094', 'max335', 'max211', 'max193', 'max171')

channel_by_decreasing_temp = ('094', '335', '211', '193', '171', '131')

types_of_bv_data = ('peak',)

bv_data_ranges = {"peak": [-3600, 3600], "max": [-1, 1]}

bv_ylabel = {"peak": 'time lag (seconds)', "max": 'cross correlation coefficient'}




pli_range = [1, 4]
bv_peak_range = [-2000, 2000]

# This section loads in the power law indices, as outputted by the
# analysis_spatial_distribution_common_spectral_model_parameters.py
csmp_waves = ['94', '131', '171', '193', '211', '335']

filepaths_root = ds.datalocationtools.save_location_calculator(ds.roots, [ds.corename, ds.sunlocation, ds.fits_level])['pickle']

# Get the output file paths
filepaths_filepath = os.path.join(filepaths_root, "analysis_spatial_distribution_common_spectral_model_parameters.filepaths.pkl")
f = open(filepaths_filepath, 'rb')
csmp_filepaths = pickle.load(f)
f.close()

# Find the particular filepaths we are interested in
this_parameter = 'powerlawindex'
parameter_info = dict()
for wave in waves:
    for csmp_filepath in csmp_filepaths:
        if (wave in csmp_filepath) and (this_parameter in csmp_filepath):
            f = open(csmp_filepath, 'rb')
            parameter_info[wave] = dict()
            parameter_info[wave]['title'] = pickle.load(f)
            parameter_info[wave]['subtitle'] = pickle.load(f)
            parameter_info[wave]['map'] = pickle.load(f)




def peak_value_to_timelag(value):
    """
    Convert a time lag index value to
    :param value:
    Return
    """
    return (-3600 + value*8)*u.s


keys = filepaths.keys()

anderson_ks = {}

for key in keys:
    # Get the files
    files = filepaths[key]

    # Read in the Bradshaw & Viall 2016 time lag data
    bv = readsav(os.path.expanduser(files["bv"]))

    # Power law indices
    pli = np.transpose(np.load(os.path.expanduser(files["pli"])))

    # Mask of power law indices
    plim = np.transpose(np.load(os.path.expanduser(files["plim"])))

    #
    pli_mask_lo = pli_range[0] < pli
    pli_mask_hi = pli_range[1] > pli
    pli_satisfied = np.logical_and(pli_mask_lo, pli_mask_hi)
    pli_mask = np.logical_or(plim, np.logical_not(pli_satisfied))

    # Go through all the BV data
    for bv_data_type in types_of_bv_data:
        for i, starting_channel in enumerate(channel_by_decreasing_temp[:-1]):
            bv_key = "{:s}{:s}".format(bv_data_type, starting_channel)

            # Get the results
            bv_all_results = bv[bv_key]

            # Number of channels
            nchannels = bv_all_results.shape[0]

            for j in range(1, nchannels):
                # Get the ending channel
                ending_channel = channel_by_decreasing_temp[i+j]

                # Get the results
                bv_results = bv_all_results[j, :, :]

                # If necessary convert to time in seconds
                if bv_data_type == 'peak':
                    bv_results = peak_value_to_timelag(bv_results).value

                # Mask out the low and hi end excessions
                bv_result_mask_lo = bv_results < bv_peak_range[0]
                bv_result_mask_hi = bv_results > bv_peak_range[1]
                bv_result_mask_abs = np.abs(bv_results) < 200
                bv_results_finite = ~np.isfinite(bv_results)

                bv_result_satisfied = np.logical_or(bv_result_mask_lo, bv_result_mask_hi)
                bv_result_satisfied = np.logical_or(bv_result_satisfied, bv_result_mask_abs)
                bv_result_mask = np.logical_or(bv_result_satisfied, bv_results_finite)

                mask = np.logical_or(pli_mask, bv_result_mask)

                # Mask using the same mask as the power law indices
                bv_data = np.ma.array(bv_results, mask=mask)
                pli_data = np.ma.array(pli, mask=mask)

                # Select the range values
                bv_data_range = bv_data_ranges[bv_data_type]

                # Make a plot
                plt.close('all')

                fig, ax = plt.subplots()
                pli_comp = pli_data.compressed()
                pli_mean = np.mean(pli_comp)
                pli_std = np.std(pli_comp)
                pli_median = np.median(pli_comp)
                pli_mad = np.median(np.abs(pli_comp-pli_median))

                bv_comp = bv_data.compressed()
                bv_mean = np.mean(bv_comp)
                bv_std = np.std(bv_comp)
                bv_median = np.median(bv_comp)
                bv_mad = np.median(np.abs(bv_comp-bv_median))

                if starting_channel == '193' and ending_channel == '171':
                    anderson_ks[key] = {"pli": pli_comp, "bv": bv_comp}

                _, _, _, cax = ax.hist2d(pli_comp, bv_comp, range=[pli_range, bv_data_range], bins=(50, 150))
                ax.axhline(bv_mean, color='red', linestyle=":", linewidth=2, label='{:.0f} ({:.0f}) s'.format(bv_median, bv_mad))
                ax.axvline(pli_mean, color='red', linewidth=2, label='{:.2f} ({:.2f})'.format(pli_median, pli_mad))
                npix = len(pli_comp)
                ax.set_xlabel('171 power law index')
                ylabel = "{:s}, {:s} - {:s}".format(bv_ylabel[bv_data_type], starting_channel, ending_channel)
                ax.set_ylabel(ylabel)
                ax.set_title(key + '\n[{:n} pixels of {:n} used]'.format(npix, mask.size))
                cbar = fig.colorbar(cax)
                cbar.ax.set_ylabel('number of pixels')
                ax.set_ylim(0, 2000)
                plt.legend(fontsize=15, framealpha=0.8)
                img_filename = 'nanoflare={:s}_{:s}_start={:s}_end={:s}_plim_171.png'.format(key, bv_data_type, starting_channel, ending_channel)
                img_filepath = os.path.join(img_save, img_filename)
                plt.savefig(img_filepath)

for key1 in keys:
    for key2 in keys:
        pli1 = anderson_ks[key1]["pli"]
        pli2 = anderson_ks[key2]["pli"]
        bv1 = anderson_ks[key1]["bv"]
        bv2 = anderson_ks[key2]["bv"]

        pli_anderson = anderson_ksamp([pli1, pli2])
        print(key1, key2, pli_anderson)

        bv_anderson = anderson_ksamp([bv1, bv2])
        print(key1, key2, bv_anderson)
