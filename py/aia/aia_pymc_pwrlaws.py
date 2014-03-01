#
# Code to fit power laws to some data from the region analysis
#
# Also do posterior predictive checking to find which model fits best.
#
import os
import pickle
import numpy as np

import pymc
from matplotlib import pyplot as plt

import aia_specific
import pymcmodels

plt.ion()
#
# Set up which data to look at
#
dataroot = '~/Data/AIA/'
ldirroot = '~/ts/pickle/'
sfigroot = '~/ts/img/'
scsvroot = '~/ts/csv/'
corename = '20120923_0000__20120923_0100'
sunlocation = 'disk'
fits_level = '1.5'
waves = ['171']
regions = ['moss', 'qs', 'loopfootpoints', 'sunspot']
windows = ['hanning']
manip = 'relative'

nsample = 1000

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

            for obstype in ['.logiobs', '.iobs']:
                #
                print('Analyzing ' + obstype)

                # Load the data
                pkl_location = locations['pickle']
                ifilename = 'OUT.' + region_id + obstype
                pkl_file_location = os.path.join(pkl_location, ifilename + '.pickle')
                print('Loading ' + pkl_file_location)
                pkl_file = open(pkl_file_location, 'rb')
                freqs = pickle.load(pkl_file)
                pwr = pickle.load(pkl_file)
                pkl_file.close()
    
                # Normalize the frequency
                xnorm = freqs[0]
                x = freqs / xnorm
    
                if obstype == '.logiobs':
                    # Set the Gaussian width
                    sigma = 0.5 * np.log(10.0)

                    # Set up the first model power law
                    pymcmodel = pymcmodels.Log_splwc(x, pwr, sigma)
        
                    # Initiate the sampler
                    M = pymc.MCMC(pymcmodel)
        
                    # Run the sampler
                    print('Running simple power law model')
                    M.sample(iter=50000, burn=10000, thin=5, progress_bar=True)

                    # Set up the second model - single power law with constant and Gaussian bump
                    pymcmodel2 = pymcmodels.Log_splwc_GaussianBump(x, pwr, sigma)
        
                    # Initiate the sampler
                    M2 = pymc.MCMC(pymcmodel2)
        
                    # Run the sampler
                    print('Running power law plus Gaussian model')
                    M2.sample(iter=50000, burn=10000, thin=5, progress_bar=True)

   
                if obstype == '.iobs':
                    # Set the Gaussian width
                    sigma = 0.5 * np.log(10.0)

                    # Set up the first model power law
                    pymcmodel = pymcmodels.Log_splwc_lognormal(x, np.exp(pwr), sigma)
        
                    # Initiate the sampler
                    M = pymc.MCMC(pymcmodel)
        
                    # Run the sampler
                    print('Running simple power law model')
                    M.sample(iter=50000, burn=10000, thin=5, progress_bar=True)

                    # Set up the second model - single power law with constant and Gaussian bump
                    pymcmodel2 = pymcmodels.Log_splwc_GaussianBump_lognormal(x, np.exp(pwr), sigma)
        
                    # Initiate the sampler
                    M2 = pymc.MCMC(pymcmodel2)
        
                    # Run the sampler
                    print('Running power law plus Gaussian model')
                    M2.sample(iter=50000, burn=10000, thin=5, progress_bar=True)

                # Plot
                plt.figure(1)
                plt.loglog(freqs, np.exp(pwr), label='data', color='k')
                
                # Plot the power law fit
                plt.loglog(freqs, np.exp(M.stats()['fourier_power_spectrum']['mean']), label='M, mean', color = 'b')
                plt.loglog(freqs, np.exp(M.stats()['fourier_power_spectrum']['95% HPD interval'][:, 0]), label='M, 95% low', color = 'b', linestyle='-.')
                plt.loglog(freqs, np.exp(M.stats()['fourier_power_spectrum']['95% HPD interval'][:, 1]), label='M, 95% high', color = 'b', linestyle='--')
                
                # Plot the power law and bump fit
                plt.loglog(freqs, np.exp(M2.stats()['fourier_power_spectrum']['mean']), label='M2, mean', color = 'r')
                plt.loglog(freqs, np.exp(M2.stats()['fourier_power_spectrum']['95% HPD interval'][:, 0]), label='M2, 95% low', color = 'r', linestyle='-.')
                plt.loglog(freqs, np.exp(M2.stats()['fourier_power_spectrum']['95% HPD interval'][:, 1]), label='M2, 95% high', color = 'r', linestyle='--')
                plt.legend(loc=3, framealpha=0.5, fontsize=10)
                plt.xlabel('frequency (Hz)')
                plt.ylabel('power')
                plt.title(obstype)
                plt.savefig(savefig + obstype + '.model_fit_compare.pymc.png')
                plt.close('all')

                #
                # Do the posterior predictive comparison
                #
                t_lrt_distribution = []
                for r in range(0, nsample):
                    # Data for the first model
                    M_data = M.trace("predictive")[r]
                    
                    # Perform the fit
                    
                    # Get the likelihood at the best fit
                    
                    
                    # Data for the second model
                    M2_data = M2.trace("predictive")[r]
                    
                    # Perform the fit
                    
                    # Get the likelihood at the best fit
                    
                    # Calculate the T_lrt statistic and keep it
                    t_lrt_distribution.append()
                    
                #
                # Plot the T_lrt distribution along with the the value of this
                # statistic
                
                plt.savefig(savefig + obstype + '.t_lrt.png')

                    