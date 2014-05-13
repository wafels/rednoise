#
#
#
import os
import pickle
import numpy as np
import aia_specific
import rnspectralmodels
from matplotlib import pyplot as plt

from paper1 import log_10_product, s171, s193, s5min, s3min, tsDetails


plt.ioff()
plt.close('all')
#
# Set up which data to look at
#
dataroot = '~/Data/AIA/'
ldirroot = '~/ts/pickle_cc_final/'
sfigroot = '~/ts/img_cc_final/'
scsvroot = '~/ts/csv_cc_final/'
corename = 'shutdownfun3_6hr'
sunlocation = 'disk'
fits_level = '1.5'
waves = ['171', '193']
regions = ['moss', 'loopfootpoints', 'sunspot', 'qs']
windows = ['hanning']
manip = 'relative'

freqfactor = [1000, 'mHz']
savefig_format = 'png'

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
                dummy = pickle.load(pkl_file)
                dummy = pickle.load(pkl_file)
                pwr_ff = pickle.load(pkl_file)
                pkl_file.close()

                # Normalize the frequency
                xnorm = freqs[0]
                x = freqs / xnorm

                # Get the data at the center of each region

                pwr_central[region + wave] = np.exp(pwr_ff)

                #
                # Get the number of elements in the time series
                #
                loc = os.path.join(os.path.expanduser(scsv), window, manip)


sunday_name = {'qs': 'quiet Sun',
               'loopfootpoints': 'loop foot points',
               'sunspot': 'sunspot',
               'moss': 'moss'}

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

    #yformatter = plt.FuncFormatter(log_10_product)
    #ax.yaxis.set_major_formatter(yformatter)

    # Geometric mean
    for region in regions:
        ax.plot(1000 * freqs, pwr_central[region + wave], label=sunday_name[region],
                linestyle=SS[wave].linestyle,
                linewidth=SS[wave].linewidth)

    if wave == '171':
        A1 = [1.1031059689844873, 1.8623857178233456, -9.140960370407038, -3.1279186484485546, 3.1658463991730077, 0.6950910275985983]

        # Calculate the best fit Gaussian contribution
        normalbump_BF = np.log(rnspectralmodels.NormalBump2_CF(np.log(x), A1[3], A1[4], A1[5]))
        # Calculate the best fit power law contribution
        powerlaw_BF = np.log(rnspectralmodels.power_law_with_constant(x, A1[0:3]))

        pl = '$P(f)=0.47f^{-1.86}$'
        gf = '$G(f)=0.046 N(\ln f: -6.85, 0.71) + 10^{-3.97}$'
        ax.plot(1000 * freqs, np.exp(powerlaw_BF), label='moss: power law $P(f)$', color='b', linestyle='-.')
        ax.plot(1000 * freqs, np.exp(normalbump_BF), label='moss: lognormal $G(f)$', color='b', linestyle=':')
        ax.plot(1000 * freqs, np.exp(rnspectralmodels.Log_splwc_AddNormalBump2(x, A1)), label='moss: $P(f) + G(f)$', color='b', linestyle='--')
        plt.text(0.31, 1.1, pl)
        plt.text(0.03, 5.1, gf)

    if wave == '193':
        A1 = [1.280530967812677, 2.076966054374039, -10.089892351088762, -4.567462870970832, 3.4415369706621, 0.5756966230782082]

        # Calculate the best fit Gaussian contribution
        normalbump_BF = np.log(rnspectralmodels.NormalBump2_CF(np.log(x), A1[3], A1[4], A1[5]))
        # Calculate the best fit power law contribution
        powerlaw_BF = np.log(rnspectralmodels.power_law_with_constant(x, A1[0:3]))

        pl = '$P(f)=0.57f^{-2.09}$'
        gf = '$G(f)=0.011 N(\ln f: -6.55, 0.58) + 10^{-4.28}$'
        ax.plot(1000 * freqs, np.exp(powerlaw_BF), label='moss: power law, $P(f)$', color='b', linestyle='-.')
        ax.plot(1000 * freqs, np.exp(normalbump_BF), label='moss: lognormal, $G(f)$', color='b', linestyle=':')
        ax.plot(1000 * freqs, np.exp(rnspectralmodels.Log_splwc_AddNormalBump2(x, A1)), label='moss: $P(f) + G(f)$', color='b', linestyle='--')
        plt.text(0.31, 1.1, pl)
        plt.text(0.03, 5.1, gf)

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
    plt.ylabel('average power (arb. units)')
    plt.title('AIA ' + wave + '$\AA$')
    plt.legend(loc=3, fontsize=10, framealpha=0.5)
    plt.ylim(0.00001, 10.0)
    # Create the branches in order
    branches = [corename, sunlocation, fits_level, wave, region]

    # Data and save locations are based here
    locations = aia_specific.save_location_calculator(roots, branches)

    # set the saving locations
    sfig = locations["image"]
    print wave
    plt.savefig(sfig + '.averagepower.%s.%s' % (wave, savefig_format))
    plt.close('all')
