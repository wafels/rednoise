#
# Code to fit power laws to some data from the region analysis
#
import os
import pickle
import numpy as np

import pymc
from matplotlib import pyplot as plt

import aia_specific
import pymcmodels


#
# Set up which data to look at
#
dataroot = '~/Data/AIA/'
ldirroot = '~/ts/pickle_cc/'
sfigroot = '~/ts/img_cc/'
scsvroot = '~/ts/csv_cc/'
corename = 'shutdownfun3_6hr'
sunlocation = 'disk'
fits_level = '1.5'
waves = ['193']
regions = ['moss']
windows = ['hanning']
manip = 'relative'

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

            # Load the data
            pkl_location = locations['pickle']
            ifilename = 'OUT.' + region_id + '.logiobs'
            pkl_file_location = os.path.join(pkl_location, ifilename + '.pickle')
            print('Loading ' + pkl_file_location)
            pkl_file = open(pkl_file_location, 'rb')
            freqs = pickle.load(pkl_file)
            pwr = pickle.load(pkl_file)
            pkl_file.close()

            # Normalize the frequency
            xnorm = freqs[0]
            x = freqs / xnorm

            # Set up the first model power law
            pymcmodel = pymcmodels.single_power_law_with_constant(freqs, pwr, likelihood_type='normal', sigma=0.5)

            # Initiate the sampler
            M = pymc.MCMC(pymcmodel)

            # Run the sampler
            M.sample(iter=50000, burn=10000, thin=5, progress_bar=True)

            """
            # Set up the second model - single power law with constant and Gaussian bump
            pymcmodel2 = pymcmodels.splwc_GaussianBump(x, pwr)

            # Initiate the sampler
            M2 = pymc.MCMC(pymcmodel2)

            # Run the sampler
            M2.sample(iter=50000, burn=10000, thin=5, progress_bar=True)
            """
plt.figure(1)
plt.loglog(freqs, pwr, label='data', color='k')
plt.loglog(freqs, M.stats()['fourier_power_spectrum']['mean'], label='M, mean', color = 'b')
plt.loglog(freqs, M.stats()['fourier_power_spectrum']['95% HPD interval'][:, 0], label='M, 95% low', color = 'b', linestyle='-.')
plt.loglog(freqs, M.stats()['fourier_power_spectrum']['95% HPD interval'][:, 1], label='M, 95% high', color = 'b', linestyle='--')

plt.loglog(freqs, M2.stats()['fourier_power_spectrum']['mean'], label='M, mean', color = 'r')
plt.loglog(freqs, M2.stats()['fourier_power_spectrum']['95% HPD interval'][:, 0], label='M2, 95% low', color = 'r', linestyle='-.')
plt.loglog(freqs, M2.stats()['fourier_power_spectrum']['95% HPD interval'][:, 1], label='M2, 95% high', color = 'r', linestyle='--')
plt.legend(loc=3, framealpha=0.5, fontsize=10)
plt.xlabel('frequency (Hz)')
plt.ylabel('power')
