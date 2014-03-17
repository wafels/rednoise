#
#
#
import os
import pickle
import numpy as np
import aia_specific

from matplotlib import pyplot as plt

from paper1 import log_10_product


plt.ioff()
plt.close('all')
#
# Set up which data to look at
#
dataroot = '~/Data/AIA/'
ldirroot = '~/ts/pickle_cc/'
sfigroot = '~/ts/img_cc/'
scsvroot = '~/ts/csv_cc/'
corename = '20120923_0000__20120923_0100'
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
wv = {}
for wave in waves:
    wv[wave] = None

pwr_central = {}
for region in regions:
    pwr_central[region] = wv


for iwave, wave in enumerate(waves):
    # Which wavelength?
    print('Wave: ' + wave + ' (%i out of %i)' % (iwave + 1, len(waves)))

    # Now that the loading and saving locations are seot up, proceed with
    # the analysis.
    for iregion, region in enumerate(regions):
        # Which region
        print('Region: ' + region + ' (%i out of %i)' % (iregion + 1, len(regions)))

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
                ifilename = 'OUT.' + region_id + '.fourier_power'
                pkl_file_location = os.path.join(pkl_location, ifilename + '.pickle')
                print('Loading ' + pkl_file_location)
                pkl_file = open(pkl_file_location, 'rb')
                freqs = pickle.load(pkl_file)
                pwr_ff = pickle.load(pkl_file)
                pkl_file.close()

                # Normalize the frequency
                xnorm = freqs[0]
                x = freqs / xnorm

                # Get the data at the center of each region
                pwr_central[region][wave] = pwr_ff[0.5 * pwr_ff.shape[0],
                                                   0.5 * pwr_ff.shape[1]]

sunday_name = {'qs': 'quiet Sun',
               'loopfootpoints': 'loop foot points',
               'sunspot': 'sunspot',
               'moss': 'moss'}

class LineStyles:
    def __init__(self):
        self.color = 'b'
        self.linewidth = 1
        self.linestyle = ""


s171 = LineStyles()
s171.color = 'b'

s193 = LineStyles()
s193.color = 'r'

s5min = LineStyles()
s5min.color ='k'
s5min.linewidth = 2
s5min.linestyle = "--"

s3min = LineStyles()
s3min.color ='k'
s3min.linewidth = 2
s3min.linestyle = "-."


SS = {}
SS['171'] = s171
SS['193'] = s193

five_min = 1.0 / 300.0
three_min = 1.0 / 180.0

#
# Got all the data.  Now make the plots
#
for region in regions:
    
    ax = plt.subplot(111)

    # Set the scale type on each axis
    ax.set_xscale('log')
    #ax.set_yscale('log')

    # Set the formatting of the tick labels
    xformatter = plt.FuncFormatter(log_10_product)
    ax.xaxis.set_major_formatter(xformatter)
    ax.yaxis.set_major_formatter(xformatter)

    # Geometric mean
    for wave in waves:
        ax.plot(freqs, pwr_ff[region][wave], label=wave,
                color=SS[wave].color,
                linestyle=SS[wave].linestyle,
                linewidth=SS[wave].linewidth)

    # Extra information for the plot
    ax.axvline(five_min, label='5 mins.',
               color=SS['5 mins'].color,
               linestyle=SS['5 mins'].linestyle,
               linewidth=SS['5 mins'].linewidth)

    ax.axvline(three_min, label='3 mins.',
               color=SS['3 mins'].color,
               linestyle=SS['3 mins'].linestyle,
               linewidth=SS['3 mins'].linewidth)

    plt.xlabel('frequency (%s)' % (freqfactor[1]))
    plt.title(sunday_name[region] + ' : example power spectra')
    plt.legend(loc=3, fontsize=10, framealpha=0.5)
    plt.savefig(savefig + '.central_power.%s' % (savefig_format))
    plt.close('all')
