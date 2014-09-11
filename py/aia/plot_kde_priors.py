#
# USE THIS ONE!
#
#
# Code to fit power laws to some data from the region analysis
#
# Also do posterior predictive checking to find which model fits best.
#
import pickle
import numpy as np
import os
#from matplotlib import rc_file
#matplotlib_file = '~/ts/rednoise/py/matplotlibrc_paper1.rc'
#rc_file(os.path.expanduser(matplotlib_file))
import matplotlib.pyplot as plt

import pymc
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit

# Normality tests
from scipy.stats import shapiro, anderson
from statsmodels.graphics.gofplots import qqplot

# Distributions that can be fit to the independence coefficient
from scipy.stats import beta as beta_distrib

import aia_specific
import pymcmodels2
import rnspectralmodels
from paper1 import sunday_name, prettyprint, log_10_product, indexplot
from paper1 import csv_timeseries_write, pkl_write, fix_nonfinite, fit_details
from aia_pymc_pwrlaws_helpers import *

# Reproducible
np.random.seed(1)

plt.ioff()
#
# Set up which data to look at
#
dataroot = '~/Data/AIA/'
ldirroot = '~/ts/pickle_cc_final/'
sfigroot = '~/ts/img_cc_final_spearman/'
scsvroot = '~/ts/csv_cc_final/'
corename = 'shutdownfun3_6hr'
sunlocation = 'disk'
fits_level = '1.5'
waves = ['171', '193']
regions = ['sunspot', 'moss', 'qs', 'loopfootpoints']
windows = ['hanning']
manip = 'relative'
neighbour = 'random'

"""
#
# Limb
#
dataroot = '~/Data/AIA/'
ldirroot = '~/ts/pickle/'
sfigroot = '~/ts/img/'
scsvroot = '~/ts/csv/'
corename = 'shutdownfun6_6hr'
sunlocation = 'limb'
fits_level = '1.0'
waves = ['171']
regions = ['highlimb', 'crosslimb', 'lowlimb', 'moss', 'loopfootpoints1', 'loopfootpoints2']
windows = ['hanning']
manip = 'relative'
"""
#
# Number of posterior predictive samples to calculate
#
# Set to 1 to do a full analytical run.  Set to a high number to do a test
# run
testing = 1

nsample = 5000 / testing

# PyMC control
itera = 100000 / testing
burn = 20000 / testing
thin = 5

#
# Kernel density estimates
#
kde = {'171': {}, '193': {}}
npx = {'171': {}, '193': {}}

# Image file type
imgfiletype = 'pdf'

# Frequency scaling
freqfactor = 1000.0

# Choose the shape of the bump
bump_shape = 'lognormal'


for iwave, wave in enumerate(waves):
    print(' ')
    print('##################################################################')
    prettyprint('Data')
    # Which wavelength?
    print('Wave: ' + wave + ' (%i out of %i)' % (iwave + 1, len(waves)))

    # Now that the loading and saving locations are seot up, proceed with
    # the analysis.
    for iregion, region in enumerate(regions):
        # Which region
        prettyprint('Region')
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

                # Load the power spectrum
                pkl_location = locations['pickle']
                ifilename = 'OUT.' + region_id + obstype
                pkl_file_location = os.path.join(pkl_location, ifilename + '.pickle')
                print('Loading ' + pkl_file_location)
                pkl_file = open(pkl_file_location, 'rb')
                # Frequencies
                freqs = pickle.load(pkl_file)
                # Number of pixels used to create the average
                npixels = pickle.load(pkl_file)
                # Mean Fourier Power per frequency
                pwr_ff_mean = pickle.load(pkl_file)
                # Fitted Fourier power per frequency
                pwr_ff_fitted = pickle.load(pkl_file)
                # Histogram Peak of the Fourier power per frequency
                pwr_ff_peak = pickle.load(pkl_file)
                pkl_file.close()

                # Load the widths
                ifilename = 'OUT.' + region_id
                pkl_file_location = os.path.join(pkl_location, ifilename + '.distribution_widths.pickle')
                print('Loading ' + pkl_file_location)
                pkl_file = open(pkl_file_location, 'rb')
                # Frequencies
                freqs = pickle.load(pkl_file)
                # Standard deviation of the Fourier power per frequency
                std_dev = pickle.load(pkl_file)
                # Fitted width of the Fourier power per frequency
                fitted_width = pickle.load(pkl_file)
                pkl_file.close()

                # Load the coherence
                ifilename = 'OUT.' + region_id
                pkl_file_location = os.path.join(pkl_location, ifilename + '.coherence.' + neighbour + '.pickle')
                print('Loading ' + pkl_file_location)
                pkl_file = open(pkl_file_location, 'rb')
                freqs = pickle.load(pkl_file)
                coher = pickle.load(pkl_file)
                coher_max = pickle.load(pkl_file)
                coher_mode = pickle.load(pkl_file)
                coher_95_hi = pickle.load(pkl_file)
                coher_mean = pickle.load(pkl_file)
                pkl_file.close()

                # Something wrong with the last coherence - estimate it
                coher_95_hi[-1] = coher_95_hi[-2]
                coher_mean[-1] = coher_mean[-2]

                # Load the correlative measures
                ifilename = 'OUT.' + region_id
                pkl_file_location = os.path.join(pkl_location, ifilename + '.correlative.' + neighbour + '.pickle')
                print('Loading ' + pkl_file_location)
                pkl_file = open(pkl_file_location, 'rb')
                cc0_ds = pickle.load(pkl_file)
                cclag_ds = pickle.load(pkl_file)
                ccmax_ds = pickle.load(pkl_file)
                pkl_file.close()

                # Load the correlative measures 2
                ifilename = 'OUT.' + region_id
                pkl_file_location = os.path.join(pkl_location, ifilename + '.spearman.' + neighbour + '.npy')
                print('Loading ' + pkl_file_location)
                ccmax = np.load(pkl_file_location)

                # Calculate the KDE
                kde[wave][region] = pymcmodels2.calculate_kde(1.0 + (1.0 - ccmax) * (npixels - 1), None)
                npx[wave][region] = npixels


for iregion, region in enumerate(regions):
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

    # Update the region identifier
    region_id = '.'.join((ident, window, manip))

    # Create a location to save the figures
    savefig = os.path.join(os.path.expanduser(sfig), window, manip)
    if not(os.path.isdir(savefig)):
        os.makedirs(savefig)
    savefig = os.path.join(savefig, region_id)

    x1_grid = np.linspace(1, npx['171'][region], 1000)
    z1 = kde['171'][region](x1_grid)
    mostprobable_npx1 = x1_grid[np.argmax(z1)]

    x2_grid = np.linspace(1, npx['193'][region], 1000)
    z2 = kde['193'][region](x2_grid)
    mostprobable_npx2 = x2_grid[np.argmax(z2)]

    plt.plot(x1_grid, z1, label='171: max(pdf) at $N_{eff}=$%i' % (np.rint(mostprobable_npx1)))
    plt.plot(x2_grid, z2, label='193: max(pdf) at $N_{eff}=$%i' % (np.rint(mostprobable_npx2)))
    plt.legend()
    plt.xlabel('$N_{eff}$')
    plt.ylabel('pdf')
    plt.title('%s pdf($N_{eff}$)' % (sunday_name[region]))
    print 'Saving to ' + savefig + obstype + '.npixels_prior_kde.%s' % (imgfiletype)
    plt.savefig(savefig + obstype + '.npixels_prior_kde.%s.%s' % (neighbour, imgfiletype))
    plt.close('all')



