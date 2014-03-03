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
import rnspectralmodels

"""
#
# Curvefit the log of the power law plus constant with a Gaussian
#
def curve_fit_LogPowerLawPlusConstantGaussian(x, y, sigma, p0=None):
    A = curve_fit(aia_plaw.LogPowerLawPlusConstantGaussian, x, y, p0=p0)
    best_fit, likelihood = calc_LogPowerLawPlusConstantGaussian(x, 
                                                   A[0][0],
                                                   A[0][1],
                                                   A[0][2],
                                                   A[0][3],
                                                   A[0][4],
                                                   A[0][5],
                                                   sigma)
    return A, best_fit, likelihood

def calc_LogPowerLawPlusConstantGaussian(x, A, sigma):
    best_fit = aia_plaw.LogPowerLawPlusConstantGaussian(x,
                                                   A[0],
                                                   A[1],
                                                   A[2],
                                                   A[3],
                                                   A[4],
                                                   A[5])
    likelihood = pymc.normal_like(best_fit, y, 1.0 / (sigma ** 2))

    return best_fit, likelihood


def calc_likelihood_gaussian(a, b, sigma):
    return pymc.normal_like(a, b, 1.0 / (sigma ** 2))


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
"""

def get_variables_M0(m):
    A = [m.value.power_law_norm, m.value.power_law_index, m.value.background]
    return A


def get_variables_M1(m):
    A = [m.value.power_law_norm, m.value.power_law_index, m.value.background,
         m.value.gaussian_amplitude, m.value.gaussian_position, m.value.gaussian_width]
    return A


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

nsample = 10

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
                    tau = 1.0 / (sigma ** 2)

                    # Set up the first model power law
                    pymcmodel0 = pymcmodels.Log_splwc(x, pwr, sigma)
        
                    # Initiate the sampler
                    M0 = pymc.MCMC(pymcmodel0)
        
                    # Run the sampler
                    print('Running simple power law model')
                    M0.sample(iter=50000, burn=10000, thin=5, progress_bar=True)
                    
                    # Now run the MAP
                    map_M0 = pymc.MAP(pymcmodel0)
                    map_M0.fit(method='fmin_powell')
                    
                    # Get the likelihood at the maximum
                    likelihood0_data = map_M0.logp_at_max
                    A0 = get_variables_M0(map_M0)
                    l0_data2 = pymc.normal_like(pwr, rnspectralmodels.Log_splwc(x, A0), tau)
                    print likelihood0_data, l0_data2
                    # Get the variables
                    #A0 = get_variables_model0(map_M0)

                    # Set up the second model - single power law with constant and Gaussian bump
                    pymcmodel1 = pymcmodels.Log_splwc_GaussianBump(x, pwr, sigma)
        
                    # Initiate the sampler
                    M1 = pymc.MCMC(pymcmodel1)
        
                    # Run the sampler
                    print('Running power law plus Gaussian model')
                    M1.sample(iter=50000, burn=10000, thin=5, progress_bar=True)
                    
                    # Now run the MAP
                    map_M1 = pymc.MAP(pymcmodel1)
                    map_M1.fit(method='fmin_powell')
                    
                    # Get the likelihood at the maximum
                    likelihood1_data = map_M1.logp_at_max
                    A1 = get_variables_M1(map_M1)
                    l1_data2 = pymc.normal_like(pwr, rnspectralmodels.Log_splwc_GaussianBump(x, A1), tau)
                    print likelihood1_data, l1_data2

                    # Get the T_lrt statistic for the data
                    t_lrt_data = likelihood0_data - likelihood1_data
    
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
                    M1 = pymc.MCMC(pymcmodel2)
        
                    # Run the sampler
                    print('Running power law plus Gaussian model')
                    M1.sample(iter=50000, burn=10000, thin=5, progress_bar=True)

                # Plot
                plt.figure(1)
                plt.loglog(freqs, np.exp(pwr), label='data', color='k')
                
                # Plot the power law fit
                plt.loglog(freqs, np.exp(M0.stats()['fourier_power_spectrum']['mean']), label='power law, mean', color = 'b')
                plt.loglog(freqs, np.exp(M0.stats()['fourier_power_spectrum']['95% HPD interval'][:, 0]), label='power law, 95% low', color = 'b', linestyle='-.')
                plt.loglog(freqs, np.exp(M0.stats()['fourier_power_spectrum']['95% HPD interval'][:, 1]), label='power law, 95% high', color = 'b', linestyle='--')
                
                # Plot the power law and bump fit
                plt.loglog(freqs, np.exp(M1.stats()['fourier_power_spectrum']['mean']), label='power law + Gaussian, mean', color = 'r')
                plt.loglog(freqs, np.exp(M1.stats()['fourier_power_spectrum']['95% HPD interval'][:, 0]), label='power law + Gaussian, 95% low', color = 'r', linestyle='-.')
                plt.loglog(freqs, np.exp(M1.stats()['fourier_power_spectrum']['95% HPD interval'][:, 1]), label='power law + Gaussian, 95% high', color = 'r', linestyle='--')
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
                M0_logp = []
                M1_logp = []
                for r in range(0, nsample):
                    # 
                    print r, nsample
                    if obstype == '.logiobs':
                        #
                        # Do the first model
                        #
                        np.random.seed(seed=1)
                        rthis = np.random.randint(0, high=8000)

                        # Generate test data under hypothesis 0
                        pred = M0.trace('predictive')[rthis]
                        
                        # Fit the test data using hypothesis 0 and calculate a likelihood
                        M0_pp = pymcmodels.Log_splwc(x, pred, sigma)
                        map_M0_pp = pymc.MAP(M0_pp)
                        map_M0_pp.fit(method='fmin_powell')
                        A0 = get_variables_M0(map_M0_pp)
                        l0_pred2 = pymc.normal_like(pwr, rnspectralmodels.Log_splwc(x, A0), tau)
                        print '**', l0_pred2
                        #bf = rnspectralmodels.Log_splwc(x,
                        #                                [map_M0_pp.value.power_law_norm,
                        #                                map_M0_pp.value.power_law_index,
                        #                                map_M0_pp.value.background])

                        # Fit the test data using hypothesis 1 and calculate a likelihood
                        M1_pp = pymcmodels.Log_splwc_GaussianBump(x, pred, sigma)
                        map_M1_pp = pymc.MAP(M1_pp)
                        map_M1_pp.fit(method='fmin_powell')
                        A1 = get_variables_M1(map_M1_pp)
                        l1_pred2 = pymc.normal_like(pwr, rnspectralmodels.Log_splwc_GaussianBump(x, A1), tau)
                        print '**', l1_pred2

                    M0_logp.append(map_M0_pp.logp_at_max)
                    M1_logp.append(map_M1_pp.logp_at_max)
                    print map_M0_pp.logp_at_max, map_M1_pp.logp_at_max, map_M0_pp.logp_at_max - map_M1_pp.logp_at_max
                    t_lrt_distribution.append(map_M0_pp.logp_at_max - map_M1_pp.logp_at_max)
                    
                #
                # Plot the T_lrt distribution along with the the value of this
                # statistic
                
                #plt.savefig(savefig + obstype + '.t_lrt.png')

t_lrt_distribution = np.asarray(t_lrt_distribution)


pvalue = np.sum(t_lrt_distribution >= t_lrt_data) / (1.0 * nsample)

plt.figure(2)
plt.hist(t_lrt_distribution, bins=100)
plt.axvline(t_lrt_data, color='k')
plt.text(t_lrt_data, 1000, 'p=%4.3f' % (pvalue))
plt.legend()
