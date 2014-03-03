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
from scipy.optimize import curve_fit

import aia_specific
import aia_plaw
import pymcmodels

#
# Curvefit the log of the power law plus constant with a Gaussian
#
def curve_fit_LogPowerLawPlusConstantGaussian(x, y, sigma, p0=None):
    A = curve_fit(aia_plaw.LogPowerLawPlusConstantGaussian, x, y, p0=p0)
    best_fit = aia_plaw.LogPowerLawPlusConstantGaussian(x,
                                                   A[0][0],
                                                   A[0][1],
                                                   A[0][2],
                                                   A[0][3],
                                                   A[0][4],
                                                   A[0][5])
    likelihood = pymc.normal_like(best_fit, y, 1.0 / (sigma ** 2))

    return A, best_fit, likelihood


#
# Curvefit the log of the power law plus constant with a Gaussian
#
def curve_fit_LogPowerLawConstant(x, y, sigma, p0=None):
    A = curve_fit(aia_plaw.LogPowerLawPlusConstant, x, y, p0=p0)
    best_fit = aia_plaw.LogPowerLawPlusConstant(x,
                                                   A[0][0],
                                                   A[0][1],
                                                   A[0][2])
    likelihood = pymc.normal_like(best_fit, y, 1.0 / (sigma ** 2))

    return A, best_fit, likelihood


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
regions = ['moss']
windows = ['hanning']
manip = 'relative'

nsample = 100

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

            for obstype in ['.logiobs']:
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
                    
                    # Get the mean parameters as a good guess.
                    Mstats = M.stats()
                    p_orig = np.asarray([np.exp(Mstats['power_law_norm']['mean']),
                                         Mstats['power_law_index']['mean'],
                                         np.exp(Mstats['background']['mean'])])
                    
                    #
                    _, _, likelihood_best = curve_fit_LogPowerLawConstant(x, pwr, sigma, p0=p_orig)

                    # Set up the second model - single power law with constant and Gaussian bump
                    pymcmodel2 = pymcmodels.Log_splwc_GaussianBump(x, pwr, sigma)
        
                    # Initiate the sampler
                    M2 = pymc.MCMC(pymcmodel2)
        
                    # Run the sampler
                    print('Running power law plus Gaussian model')
                    M2.sample(iter=50000, burn=10000, thin=5, progress_bar=True)
                    M2stats = M2.stats()
                    p2_orig = np.asarray([np.exp(M2stats['power_law_norm']['mean']),
                                         M2stats['power_law_index']['mean'],
                                         np.exp(M2stats['background']['mean']),
                                         M2stats['gaussian_amplitude']['mean'],
                                         M2stats['gaussian_position']['mean'],
                                         M2stats['gaussian_width']['mean']])
                    
                    #
                    _, _, likelihood2_best = curve_fit_LogPowerLawPlusConstantGaussian(x, pwr, sigma, p0=p2_orig)

   
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
                    
                    # Get some mean paramters

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
                    print r
                    #
                    # Data for the first model
                    #
                    M_pred = M.trace("predictive")[r]
                    
                    # Perform the fit
                    if obstype == '.logiobs':
                        # Perform the fit on the posterior data
                        A, bf, likelihood = curve_fit_LogPowerLawConstant(x, M_pred, sigma, p0=p_orig)

                    # ---------------------------------------------------------
                    #
                    # Data for the second model
                    #
                    # Get the data, find the MAP fit
                    M2_pred = M2.trace("predictive")[r]

                    if obstype == '.logiobs':
                        # Perform the fit on the posterior data
                        A2, bf2, likelihood2 = curve_fit_LogPowerLawPlusConstantGaussian(x, M2_pred, sigma, p0=p2_orig)


                    # Calculate the T_lrt statistic and keep it
                    print likelihood, likelihood2
                    t_lrt_distribution.append(likelihood - likelihood2)
                    
                #
                # Plot the T_lrt distribution along with the the value of this
                # statistic
                
                #plt.savefig(savefig + obstype + '.t_lrt.png')

                    