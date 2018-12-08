import os
import pickle
from copy import deepcopy
import numpy as np
from scipy.io import readsav
from scipy.stats import anderson_ksamp
import matplotlib.pyplot as plt
import astropy.units as u
from sunpy.map import Map
import details_study as ds

# This section loads in the VK time lag data
# Where the data is
filepaths = {"vk": "~/Data/ts/vk/crosscor06_19_12hr.sav",
             "vk193replacement": "~/Data/ts/vk/crosscor06_19_12hr_0_193.sav"}

# Load in the data
vk = readsav(os.path.expanduser(filepaths["vk"]))
vk193replacement = readsav(os.path.expanduser(filepaths["vk193replacement"]))
vk['peak193'] = deepcopy(vk193replacement['peak193'])
vk['max193'] = deepcopy(vk193replacement['max193'])

# Ordered data
#vk_keys = ('peak335', 'peak211', 'peak193', 'peak171', 'peak094', 'max094', 'max335', 'max211', 'max193', 'max171')

vk_keys = {'peak': ('peak094', 'peak335', 'peak211', 'peak193', 'peak171'),
           'max': ('max094', 'max335', 'max211', 'max193', 'max171')}


# The VK temperature order
channel_by_decreasing_temp = ('094', '335', '211', '193', '171', '131')

# The VK data we are looking at
type_of_vk_data = 'peak'

vk_data_ranges = {"peak": [-10000, 10000], "max": [-1, 1]}

vk_ylabel = {"peak": 'time lag (seconds)', "max": 'cross correlation coefficient'}


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
waves = ('94', '335', '211', '193', '171', '131')
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

pli_range = [0, 2]
vk_range = ([-1, 1]*u.h).to(u.s).value


def peak_value_to_timelag(value):
    """
    Convert a time lag index value to
    :param value:
    Return
    """
    dx = 24 * u.h / 500
    return (-12 * u.h + value * dx).to(u.s).value


anderson_ks = {}

vk_x_range = ds.vk_x_range.value
vk_y_range = ds.vk_y_range.value

ar_y = ds.ar_x
ar_x = ds.ar_y


mask_type = 'index only'
mask_type = 'also exclude AR'
mask_type = 'also exclude outlying'

mask_types = ('index only', 'also exclude AR', 'also exclude outlying')

for mask_type in mask_types:

    for wave in waves:

        # The power law index measurement
        pli_measurement = 'n (channel={:s} mask={:s})'.format(wave, mask_type)

        for vk_key in vk_keys[type_of_vk_data]:
            #
            print('Measurement type and initial channel = {:s}'.format(vk_key))
            #
            filename = ''.format(vk_key)
            vk_data = vk[vk_key][:, int(vk_y_range[0]):int(vk_y_range[1]), int(vk_x_range[0]):int(vk_x_range[1])]

            if 'peak' in vk_key:
                vk_data = peak_value_to_timelag(vk_data)

            # Number of channels
            nchannels = vk_data.shape[0]
            #
            this_map = parameter_info[wave]['map']

            # Very rough AR mask
            exclude_ar = np.zeros_like(this_map.mask)
            exclude_ar[int(ar_y[0].value):int(ar_y[1].value), int(ar_x[0].value):int(ar_x[1].value)] = True

            # Mask
            if mask_type == 'index only':
                mask = this_map.mask
            elif mask_type == 'also exclude AR':
                mask = np.logical_or(this_map.mask, exclude_ar)
            elif mask_type == 'also exclude outlying':
                mask = np.logical_or(this_map.mask, np.logical_not(exclude_ar))

            # Power law index data
            pli_data_masked = np.ma.array(this_map.data, mask=mask)
            pli_comp = pli_data_masked.compressed()

            pli_mean = np.mean(pli_comp)
            pli_std = np.std(pli_comp)
            pli_median = np.median(pli_comp)
            pli_mad = np.median(np.abs(pli_comp-pli_median))

            # go through the channels
            channels_remaining = channel_by_decreasing_temp[-nchannels:]
            for j in range(1, len(channels_remaining)):
                # The VK measurement
                ending_channel = channels_remaining[j]
                print('   - ending channel is {:s}.'.format(ending_channel))
                vk_measurement = '{:s}_cc_{:s}'.format(vk_key, ending_channel)

                # Get the VK data
                vk_data_masked = np.ma.array(vk_data[j, :, :], mask=mask)
                vk_comp = vk_data_masked.compressed()

                vk_mean = np.mean(vk_comp)
                vk_std = np.std(vk_comp)
                vk_median = np.median(vk_comp)
                vk_mad = np.median(np.abs(vk_comp-vk_median))

                plt.close('all')
                fig, ax = plt.subplots()

                _, _, _, cax = ax.hist2d(pli_comp, vk_comp, range=(pli_range, vk_range), bins=(50, 50))
                ax.axhline(vk_mean, color='red', linestyle=":", linewidth=2, label='{:.0f} ({:.0f}) s'.format(vk_mean, vk_std))
                ax.axvline(pli_mean, color='red', linewidth=2, label='{:n} ({:n})'.format(pli_mean, pli_std))
                npix = len(pli_comp)
                ax.set_xlabel('{:s} power law index'.format(wave))
                ylabel = "{:s} to {:s}".format(vk_key, ending_channel)
                ax.set_ylabel(ylabel)
                ax.set_ylim(vk_range)
                ax.set_title('{:n} pixels of {:n} used\n{:s}'.format(npix, mask.size, mask_type))
                cbar = fig.colorbar(cax)
                cbar.ax.set_ylabel('number of pixels')
                plt.legend(fontsize=15, framealpha=0.8)
                img_filename = 'nanoflare={:s}_versus_{:s}.png'.format(pli_measurement, vk_measurement)
                img_filepath = os.path.join(filepaths_root, img_filename)
                plt.tight_layout()
                plt.savefig(img_filepath)
                #print('Saved {:s}'.format(img_filepath))


stop
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
    for bv_data_type in types_of_vk_data:
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
                bv_data_range = vk_data_ranges[bv_data_type]

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
