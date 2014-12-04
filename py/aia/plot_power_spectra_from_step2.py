#
#
#
import os
from matplotlib import rc_file
matplotlib_file = '~/ts/rednoise/py/matplotlibrc_paper1.rc'
rc_file(os.path.expanduser(matplotlib_file))
import matplotlib.pyplot as plt
import pickle
import numpy as np
import aia_specific

from paper1 import log_10_product, s171, s193, s5min, s3min, sunday_name, figure_example_ps


plt.ioff()
plt.close('all')
#
# Set up which data to look at
#
dataroot = '~/Data/AIA/'
ldirroot = '~/ts/pickle_cc_False_dr_False/'
sfigroot = '~/ts/img_cc_False_dr_False/'
scsvroot = '~/ts/csv_cc_False_dr_False/'
corename = 'study2'
sunlocation = 'equatorial'
fits_level = '1.5'
waves = ['171']
regions = ['R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7']
windows = ['hanning']
manip = 'relative'

freqfactor = [1000, 'mHz']
savefig_format = 'png'


five_min = freqfactor[0] * 1.0 / 300.0
three_min = freqfactor[0] * 1.0 / 180.0

coherence_wsize = 3



#
# Storage for the power at the center of each region
#
pwr_central = {}


for iregion, region in enumerate(regions):
    # Which region
    print('Region: ' + region + ' (%i out of %i)' % (iregion + 1, len(regions)))

    for iwave, wave in enumerate(waves):
        # Which wavelength?
        print('Wave: ' + wave + ' (%i out of %i)' % (iwave + 1, len(waves)))

    # Now that the loading and saving locations are seot up, proceed with
    # the analysis.

        # Create the branches in order
        branches = [corename, sunlocation, fits_level, wave, region]

        # Set up the roots we are interested in
        roots = {"pickle": ldirroot,
                 "image": sfigroot,
                 "csv": scsvroot}

        # Data and save locations are based here
        locations = aia_specific.save_location_calculator(roots,
                                     branches)

        # set the saving locations
        sfig = locations["image"]
        scsv = locations["csv"]

        # Identifier
        ident = aia_specific.ident_creator(branches)

        # Go through all the windows
        for iwindow, window in enumerate(windows):
            # Which window
            print('Window: ' + window + ' (%i out of %i)' % (iwindow + 1, len(windows)))

            # Update the region identifier
            region_id = '.'.join((ident, window, manip))

            # Create a location to save the figures
            savefig = os.path.join(os.path.expanduser(sfig), window, manip)
            if not(os.path.isdir(savefig)):
                os.makedirs(savefig)
            savefig = os.path.join(savefig, region_id)

            #for obstype in ['.iobs', '.logiobs']:
            for obstype in ['.logiobs']:
                #
                print('Analyzing ' + obstype)
                if obstype == '.iobs':
                    print('Data is lognormally distributed.')
                else:
                    print('Data is normally distributed.')

                # Load the data
                pkl_location = locations['pickle']
                ifilename = 'OUT.' + region_id + '.logiobs'
                pkl_file_location = os.path.join(pkl_location, ifilename + '.pickle')
                print('Loading ' + pkl_file_location)
                pkl_file = open(pkl_file_location, 'rb')
                freqs = pickle.load(pkl_file)
                npixels = pickle.load(pkl_file)
                logiobs = pickle.load(pkl_file)
                logiobs_peak_fitted = pickle.load(pkl_file)
                logiobs_peak = pickle.load(pkl_file)
                pkl_file.close()

                # Normalize the frequency
                xnorm = freqs[0]
                x = freqs / xnorm

                # Get the data we are analyzing
                pwr_central[region + wave] = np.exp(logiobs)


SS = {}
SS['171'] = s171
SS['193'] = s193
SS['5 mins'] = s5min
SS['3 mins'] = s3min

five_min = 1.0 / 300.0
three_min = 1.0 / 180.0

#
# Got all the data.  Now make the plots
#
for wave in waves:

    ax = plt.subplot(111)

    # Set the scale type on each axis
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Set the formatting of the tick labels

    xformatter = plt.FuncFormatter(log_10_product)
    ax.xaxis.set_major_formatter(xformatter)

    # Geometric mean
    for i, region in enumerate(regions):
        if i >= 7:
            linestyle = ":"
        else:
            linestyle = "-"
        ax.plot(1000 * freqs, pwr_central[region + wave], label=region,
                linestyle=linestyle,
                linewidth=SS[wave].linewidth)

    # Extra information for the plot
    ax.axvline(1000 * five_min, label='5 mins.',
               color=SS['5 mins'].color,
               linestyle=SS['5 mins'].linestyle,
               linewidth=SS['5 mins'].linewidth)

    ax.axvline(1000 * three_min, label='3 mins.',
               color=SS['3 mins'].color,
               linestyle=SS['3 mins'].linestyle,
               linewidth=SS['3 mins'].linewidth)

    plt.xlabel('frequency (%s)' % (freqfactor[1]))
    plt.ylabel('power (arb. units)')
    plt.title(wave + r'$\AA$')
    plt.legend(fontsize=14, framealpha=0.5)

    # Create the branches in order
    branches = [corename, sunlocation, fits_level, wave, region]

    # Data and save locations are based here
    locations = aia_specific.save_location_calculator(roots, branches)

    # set the saving locations
    sfig = locations["image"]

    plt.savefig(sfig + '.data.%s' % (savefig_format))
    plt.close('all')
